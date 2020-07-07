
tic
rng(0)
nn=0;
load('60000trainimgsMNIST.mat')
load('trainlabels.mat')

nummaps_1=4;%4  ;%  4/6;%16;
% load('maps1nm8negw')%nm8
% load('aermaps1nm6th10t40')%nm6
% maps_1 of size 5x5

% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % load('maps1_nm6_negw')
% load('maps_1nm6_10c_negw.mat')
% load('nm7_t50_maps_1')
% load('maps1_t15th10nm16')
% load('maps_1_aer4nmth10t100')
maps_1 = zeros(5,5,4);
maps_1(:,:,1)= [1 1 1 0 0;1 1 1 0 0;1 1 1 0 0;1 1 1 0 0;1 1 1 0 0];
maps_1(:,:,2)= [1 1 1 1 1;1 1 1 1 1;1 1 1 1 1;0 0 0 0 0;0 0 0 0 0];
maps_1(:,:,3)= [1 1 0 0 0;1 1 1 0 0;0 1 1 1 0;0 0 1 1 1;0 0 0 1 1];
maps_1(:,:,4) = fliplr(maps_1(:,:,3));
% maps_1(maps_1==0)=-1;

nummaps_2 = 500;
npc=50;%numperclass
% maps_2=double(0.8.*ones(14,14,nummaps_1,nummaps_2));
% % maps_2parallel=double(.8.*ones(14,14,nummaps_1,npc,10));

% maps_2parallel=double(.8.*ones(14,14,nummaps_1,npc,10));
maps_2parallel=double(normrnd(.8,.05,[14,14,nummaps_1,npc,10]));
% maps_2parallel=double(.8.*ones(13,13,nummaps_1,npc,10));
% % % maps_2parallel=double(.8.*ones(11,11,nummaps_1,npc,10));
% % % maps_2parallel=double(.8.*ones(13,13,nummaps_1,npc,10));
% % % maps_2parallel=double(.8.*ones(5,5,nummaps_1,npc,10));

thresh_1= 10;
% thresh_2= 20.*ones(nummaps_2,1);
thresh_2parallel= 20.*ones(npc,10);

% DoG_inner = double(fspecial('gaussian',7,1) - fspecial('gaussian',7,2));
DoG_outer = double(fspecial('gaussian',7,2) - fspecial('gaussian',7,1));

timeperimg=30;
% timecut = 50 ;%ceil(timeperimg*.9);
% timelearn = timecut.*ones(10,1);
imgbinthresh=50;


numEps = 3;

% data_negW_nm4nm600_t40_tc24_thvarminhalf_4ep




opspk2=zeros(numEps*6750,10);
opspkcnt2=ones(10,1);
[inputsDim1, inputsDim2]=size(trainimgs(:,:,1));
thresh2_evol = zeros(6750*numEps,npc,10);
evtimearray=zeros(6750*numEps,10);
convergence=zeros(6750*numEps,10);
minthresh=zeros(npc,10);

firstneged=zeros(npc,10);
timeoffirstnegw=zeros(npc,10);

boundconst=20; % 1 is fast, 4 is normal, 10 is slow, match 10 with faster lr to balance. 1-> 1 at 0, 4 -> .25 at 0
% 400 = 40*10
tic


%
% trimgs=zeros(28,28,numEps*60000);
% trlbls=zeros(numEps*60000,1);
% for i = 1:numEps
%     trimgs(:,:,((i-1)*60000)+1:i*60000)=trainimgs;
%     trlbls(((i-1)*60000)+1:i*60000)=trainlabels;
% end




parfor classes = 0:9
    %     data = eval(['data',num2str(classes)]);
    %     data = trimgs(:,:,trlbls==classes);%
    currmaps_2 = maps_2parallel(:,:,:,:,classes+1);
    currthresh_2 = thresh_2parallel(:,classes+1);
    %     maps_1 = maps_1parallel(:,:,:,classes+1);
    
    currevtimearray = evtimearray(:,classes+1);
    currthresh2_evol = thresh2_evol(:,:,classes+1);
    curropspk2 = opspk2(:,classes+1);
    curropspkcnt2 = opspkcnt2(classes+1);
    currconvergence = convergence(:,classes+1);
    currfirstneged= firstneged(:,classes+1);
    currtimeoffirstnegw = timeoffirstnegw(:,classes+1);
    
    currminthresh = minthresh(:,classes+1);
    
    
    
    currdata = trainimgs(:,:,trainlabels==classes );
    for inputnum = 1:size(currdata,3)*numEps
        
        
        %     for inputnum = 1:size(data,3)
        
        
        
        %         aplus = 1;%lr_aplus(ceil(inputnum/60000));
        %         aminusno = 1;%lr_aminus(ceil(inputnum/60000));
        %         aminuspost = 1;%lr_aminuspost(ceil(inputnum/60000));
        
        %         currlabel = classes;%double(trainlabels(mod(inputnum-1,60000)+1));
        
        %     img = trainimgs(:,:,mod(inputnum-1,60000)+1);
        %         img = data(:,:,(inputnum));
        img = currdata(:,:,mod(inputnum-1,size(currdata,3))+1);
        
        %         conv_img = imfilter(double(img>=imgbinthresh),DoG_inner);
        conv_img = imfilter(double(img>=imgbinthresh),DoG_outer);
        
        maxval=max(conv_img(:));
        conv_img=max(0,conv_img);
        
        neuroncoding = (timeperimg+1) - round(((conv_img)*(timeperimg)/(maxval)));
        
        %
        count_ae=0;
        addressEvent=struct();
        for  time= 1:timeperimg
            spikes = neuroncoding==time;
            
            if sum(sum(spikes))~=0
                count_ae = count_ae+1;
                addressEvent(count_ae).numEvents=sum(sum(spikes));
                [addressEvent(count_ae).event_x,addressEvent(count_ae).event_y]=find(spikes);
                addressEvent(count_ae).eventTime=time;
                
            end
        end
        %
        
        potential=zeros(inputsDim1,inputsDim2,nummaps_1);
        hasspiked_pool = zeros(14,14,nummaps_1);
        aerpool = struct();
        count_pool = 0;
        level2_spikes = zeros(14,14,nummaps_1);
        stcnt=0;
        for k=1:timeperimg
            evTime=k;
            currevents = addressEvent([addressEvent.eventTime]==k);
            if ~isempty(currevents)
                stcnt=stcnt+1;
                
                for evs = 1:currevents.numEvents
                    xevent = currevents.event_x(evs);
                    yevent = currevents.event_y(evs);
                    
                    pool_x = ceil(xevent/2);
                    pool_y = ceil(yevent/2);
                    
                    
                    for x = max(3,xevent-2):min(xevent+2,inputsDim1-2)
                        for y = max(3,yevent-2):min(yevent+2,inputsDim2-2)
                            for n = 1:nummaps_1
                                
                                pot = potential(x,y,n);
                                potential(x,y,n) = pot + maps_1(xevent-x+3,yevent-y+3,n);
                            end
                        end
                    end
                end
            end
            while (max(potential(:))>=thresh_1)
                [win_x,win_y,win_n] = ind2sub(size(potential),find(potential==max(potential(:)),1));
                pool_x = ceil(win_x/2);
                pool_y = ceil(win_y/2);
                if hasspiked_pool(pool_x,pool_y,win_n)==1
                    potential(win_x,win_y,win_n)=0;
                    
                elseif hasspiked_pool(pool_x,pool_y,win_n)==0
                    count_pool = count_pool+1;
                    aerpool(count_pool).event_x=pool_x;
                    aerpool(count_pool).event_y=pool_y;
                    aerpool(count_pool).win_n=win_n;
                    aerpool(count_pool).eventTime=evTime;
                    
                    
                    level2_spikes(pool_x,pool_y,win_n)=evTime;
                    
                    hasspiked_pool(pool_x,pool_y,:)=1;
                    %                     hasspiked_pool(pool_x,pool_y,win_n)=1;
                    
                    potential(win_x,win_y,:)=0;
                    %                     potential(win_x,win_y,win_n)=0;
                    
                end
            end
            
        end
        
        %         hasspiked2_curr_n = zeros(npc,1);
        pots2 = zeros(npc,1);
        %         curr_pot2=zeros(2,2,npc);
        
        
        %         numevsinloc = zeros(npc,1);
        %         crossed10 = zeros(npc,1);
        spiked=0;
        for k = 1:timeperimg
            evTime=k;
            if spiked==0
                currevents = aerpool([aerpool.eventTime]==k);
                
                
                if ~isempty(currevents)
                    for evs = 1:length(currevents)
                        
                        xevent = currevents(evs).event_x;
                        yevent = currevents(evs).event_y;
                        win_n =currevents(evs).win_n;
                        
                        % % %                     for x = max(7,xevent-6):min(xevent+6,8)
                        % % %                         for y = max(7,yevent-6):min(yevent+6,8)
                        % % %                             pot = curr_pot2(x-6,y-6,:);
                        % % %                             curr_pot2(x-6,y-6,:) = squeeze(pot) + squeeze(currmaps_2(xevent-x+7,yevent-y+7,win_n,:));
                        % % %                         end
                        % % %                     end
                        pots2=pots2+squeeze(currmaps_2(xevent,yevent,win_n,:));
                        %                     numevsinloc = numevsinloc+1;
                        %                     crossed10=numevsinloc>=10;
                    end
                end
            end
            if max(pots2)>=25 && spiked==0
                % % %         pots2=squeeze(max(max(curr_pot2,[],1),[],2));
                
                
                %         [~,win_n2]=min(abs(pots2-currthresh_2));
                %         hasspiked2_curr_n(win_n2)=1;
                
                
                [~,win_n2]=max(pots2) ; %./thresh_2);
                %         [wnx,wny] = find(curr_pot2(:,:,win_n2)==max(curr_pot2(:,:,win_n2)));
                %         win_x=wnx(1)+6;win_y=wny(1)+6;
                
                
                
                currevtimearray(curropspkcnt2)=evTime;
                curropspk2(curropspkcnt2)=win_n2;
                curropspkcnt2=curropspkcnt2+1;
                
                prespikemap = level2_spikes>0 & level2_spikes<=evTime;
                prespike_yes=prespikemap;
                % %         prespike_yes=zeros(13,13,nummaps_1);
                % %         for x =max(1,win_x-6):min(win_x+6,14)
                % %             for y =max(1,win_y-6):min(win_y+6,14)
                % %                 for n = 1:nummaps_1
                % %                     %                             x,y,n
                % %                     prespike_yes(x-win_x+7,y-win_y+7,n)=prespikemap(x,y,n);
                % %                 end
                % %             end
                % %         end
                
                %         spikemapno = level2_spikes==0 ;
                %         nospike_yes=spikemapno;
                
                
                winmap=currmaps_2(:,:,:,win_n2);
                        currmaps_2(:,:,:,win_n2) = winmap + (prespike_yes.*((1-winmap.^2)./boundconst))...
                            -((1-prespike_yes).*((1-winmap.^2)./boundconst));%...
%                 currmaps_2(:,:,:,win_n2) = winmap + (prespike_yes.*(winmap.*(1-winmap)./boundconst))...
%                     -((1-prespike_yes).*(winmap.*(1-winmap)./boundconst));%...
                %         currmaps_2(:,:,:,win_n2) = winmap + (prespike_yes.*((1-winmap)./boundconst))...
                %             -((1-prespike_yes).*((1+winmap)./boundconst));
                %         currmaps_2(:,:,:,win_n2) = winmap + (prespike_yes.*((1-winmap)./boundconst))...
                %             -((1-prespike_yes).*((1+winmap)./boundconst));%...
                
                
                currthresh2_evol(inputnum,:) = currthresh_2;
                
                if (currthresh_2(win_n2))< (pots2(win_n2))
                    diff = (pots2(win_n2)) - currthresh_2(win_n2);
                    currthresh_2(win_n2) = (currthresh_2(win_n2)+(.1.*diff));
                end
                if (currthresh_2(win_n2))> (pots2(win_n2) )
                    diff = currthresh_2(win_n2) - (pots2(win_n2));
                    currthresh_2(win_n2) = (currthresh_2(win_n2)-(.1.*diff)); %currminthresh|(win_n2) o 10
                    %             currthresh_2(win_n2) = max(1,currthresh_2(win_n2)-(.15.*diff)); %currminthresh|(win_n2) o 10
                end
                
%                 pots2=pots2.*0;
                spiked=1;
                
                if currfirstneged(win_n2)==0
                    winmp=currmaps_2(:,:,:,win_n2);
                    if any(winmp(:)<0)
                        currtimeoffirstnegw(win_n2)=inputnum;
                        currfirstneged(win_n2)=1;
                        currminthresh(win_n2)=1.*currthresh_2(win_n2);
                    end
                end
                
            end
        end
        if spiked==0
            [~,win_n2]=max(pots2) ;
            
            
            currevtimearray(curropspkcnt2)=evTime+2;
            curropspk2(curropspkcnt2)=win_n2;
            curropspkcnt2=curropspkcnt2+1;
            currthresh2_evol(inputnum,:) = currthresh_2;
            
            
            prespikemap = level2_spikes>0 & level2_spikes<=timeperimg;
            prespike_yes=prespikemap;
            
            winmap=currmaps_2(:,:,:,win_n2);
            currmaps_2(:,:,:,win_n2) = winmap + (prespike_yes.*((1-winmap.^2)./boundconst))...
                -((1-prespike_yes).*((1-winmap.^2)./boundconst));%...
%             currmaps_2(:,:,:,win_n2) = winmap + (prespike_yes.*(winmap.*(1-winmap)./boundconst))...
%                 -((1-prespike_yes).*(winmap.*(1-winmap)./boundconst));%...
            
            
            
            if (currthresh_2(win_n2))< (pots2(win_n2))
                diff = (pots2(win_n2)) - currthresh_2(win_n2);
                currthresh_2(win_n2) = (currthresh_2(win_n2)+(.1.*diff));
            end
            if (currthresh_2(win_n2))> (pots2(win_n2) )
                diff = currthresh_2(win_n2) - (pots2(win_n2));
                currthresh_2(win_n2) = (currthresh_2(win_n2)-(.1.*diff)); %currminthresh|(win_n2) o 10
                %             currthresh_2(win_n2) = max(1,currthresh_2(win_n2)-(.15.*diff)); %currminthresh|(win_n2) o 10
            end

        end
        %
        cl=0;
        for ff=1:npc
            for f =1:nummaps_1
                for g = 1:size(currmaps_2,1)*size(currmaps_2,2)
                    [xind, yind]=ind2sub([size(currmaps_2,1) size(currmaps_2,2)],g);
                    wfg = abs(currmaps_2(xind,yind,f,ff));
                    cl= cl+ (wfg .*(1-wfg));
                end
            end
        end
        currconvergence(inputnum)=cl/(ff*f.*g);
    end
    maps_2parallel(:,:,:,:,classes+1)=currmaps_2;
    thresh_2parallel(:,classes+1)=currthresh_2;
    
    opspk2(:,classes+1)=curropspk2;
    evtimearray(:,classes+1)=currevtimearray;
    thresh2_evol(:,:,classes+1)=currthresh2_evol;
    opspkcnt2(classes+1)=curropspkcnt2;
    convergence(:,classes+1)=currconvergence;
    timeoffirstnegw(:,classes+1)=currtimeoffirstnegw;
    minthresh(:,classes+1)=currminthresh;
    
    classes
end
toc


figure,
for i = 0:9
    plot((i*npc)+1:(i+1)*npc,thresh_2parallel(:,i+1),'*'),hold on,
    plot(((i)*npc)+1:(i+1)*npc,minthresh(:,i+1),'r.'),hold on, grid on
    
end


figure,
for i = 0:9
    plot((i*npc)+1:(i+1)*npc,thresh_2parallel(:,i+1),'*'),hold on,grid on
end

figure,
for i = 1:10
    subplot(5,2,i)
    conve = convergence(:,i); conve(conve==0)=[];
    plot(conve),hold on,
    %     plot(timeoffirstnegw(:,i),conve(timeoffirstnegw(:,i)),'r.')
end

figure,
for i = 1:10
    jjj=opspk2(:,i); jjj(jjj==0)=[];
    subplot(2,5,i),histogram(jjj(:),npc)
end

figure,
for i = 1:10
    jjj=evtimearray(:,i); jjj(jjj==0)=[];
    subplot(5,2,i),histogram(jjj(:),timeperimg)
end


figure, histogram(maps_2parallel(:),100)

figure,
conve = sum(convergence,2);
plot(conve)%,hold on, plot(timeoffirstnegw(:),conve(timeoffirstnegw(:)),'r.')

if 0
    sz1=size(maps_1,1);
    sz2 = size(maps_2parallel,1);
    myfig = double(zeros(sz1*sz2,sz1*sz2,nummaps_2));
    
    for c = 2
        for i = 1:npc
            [val,loc] = max(maps_2parallel(:,:,:,i,c),[],3);
            %             if min(maps_2parallel(:)<0)
            %                 val=val+1;
            %             end
            
            for xx =1:sz2
                for yy =1:sz2
                    myfig(((xx-1)*sz1)+1:((xx*sz1)),((yy-1)*sz1)+1:((yy*sz1)),i) = ...
                        val(xx,yy).*(maps_1(:,:,loc(xx,yy)));
                    
                    
                    %                     tmp=0;
                    %                     for nm=1:nummaps_1
                    %                         tmp=tmp+(maps_2parallel(xx,yy,nm,i,c).*(maps_1(:,:,nm)));
                    %                     end
                    %                     myfig(((xx-1)*sz1)+1:xx*sz1,((yy-1)*sz1)+1:(yy*sz1),i) = nummaps_1+tmp;
                    %
                end
                
                
            end
            tmpp=myfig(:,:,i);
            %         figure, imagesc(myfig(:,:,i)./max(tmpp(:))), drawnow
            figure, imagesc(myfig(:,:,i)), drawnow, %pause(0.1)
        end
    end
end

meanthreshes=[];
figure,
for i = 1:10
    meanthreshes=mean(thresh2_evol(1:opspkcnt2(i)-1,((i-1)*npc)+1:(i*npc))');
    plot(meanthreshes),hold on
    txt=['\rightarrow','Digit ',num2str(i-1)];
    text(opspkcnt2(i)-1,meanthreshes(opspkcnt2(i)-1),txt)
    meanthreshes=[];
end

% save('temp_thresh_2','thresh_2parallel')
% save('temp_maps2','maps_2parallel')

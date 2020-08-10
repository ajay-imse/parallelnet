
close all, clear all,
fileID = fopen('alltrials.txt','r');

tline = fgetl(fileID);
out=tline;
while ischar(tline)
    tline = fgetl(fileID);
    out=char(out,tline);
end


fclose(fileID);
numFiles = length(out)-1

myframes=false(64,64,100,9*numFiles);
labels=zeros(9*numFiles,1);

samplenumber=1;

for file = 1:numFiles
    file
    currfilename=out(file,:);
    labelfilename=regexprep(currfilename,'.aedat','_labels.csv');
    
    labelstruct=mfcsvread(labelfilename);
    
    aedat.importParams.filePath=out(file,:);
    aedat=ImportAedat(aedat);
    newdata=[];
    newdata=aedat.data.polarity;
    
    
    
    for actionnum=1:10
        rightnum=find(labelstruct.class==actionnum,1);
        
        if ~isempty(rightnum)
%             actionnum
            
            currtime=labelstruct.startTime_usec(rightnum);
            
            for frames = 1:100
                
                currdata_x=newdata.x(newdata.timeStamp>=currtime & newdata.timeStamp<=(currtime+40000));
                
                currdata_y=newdata.y(newdata.timeStamp>=currtime & newdata.timeStamp<=(currtime+40000));
                
                currtime=currtime+40000;
                
                
                
                
                img=zeros(128);
                
                for len=1:length(currdata_x)
                    img(currdata_y(len)+1,currdata_x(len)+1)=img(currdata_y(len)+1,currdata_x(len)+1)+1;
                end
                
                im=img;
                im_nw=im(1:2:end,1:2:end);
                im_sw=im(2:2:end,1:2:end);
                im_se=im(2:2:end,2:2:end);
                im_ne=im(1:2:end,2:2:end);
                % Select pixel with maximum intensity
                im_max=max(cat(3,im_nw,im_sw,im_se,im_ne),[],3);
                
                %             imagesc(im_max>=6),drawnow,hold on,
                
                
                spikeim=im_max>=6;
                
                myframes(:,:,frames,samplenumber)=spikeim;
                
                
                
            end
            labels(samplenumber)=actionnum;

            samplenumber=samplenumber+1;
            
        end
    end
end







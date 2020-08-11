clear all, close all


labelnames={'hand_clapping','right_hand_wave','left_hand_wave','right_hand_clockwise','right_hand_counter_clockwise','left_hand_clockwise','left_hand_counter_clockwise','forearm_roll_forward','drums','guitar'};

% labelnames=categorical(labelnames)

lblstr=load('ibmlabels.mat');
labels=lblstr.labels;

k0labels_ts=labels(915:end);
k0predsstr=load('bestresults_preds_splitnet_k0.mat');
k0preds=k0predsstr.preds;
acc0=sum(k0labels_ts==k0preds)./3.05;
conf0=confusionmat(uint8(k0preds)+1,uint8(k0labels_ts)+1);


k1labels_ts=labels(306:610);
k1predsstr=load('bestresults_preds_splitnet_k1.mat');
k1preds=k1predsstr.preds;
acc1=sum(k1labels_ts==k1preds)./3.05;
conf1=confusionmat(uint8(k1preds)+1,uint8(k1labels_ts)+1);


k2labels_ts=labels(611:915);
k2predsstr=load('bestresults_preds_splitnet_k2.mat');
k2preds=k2predsstr.preds;
acc2=sum(k2labels_ts==k2preds)./3.05;
conf2=confusionmat(uint8(k2preds)+1,uint8(k2labels_ts)+1);


k3labels_ts=labels(1:305);
k3predsstr=load('bestresults_preds_splitnet_k3.mat');
k3preds=k3predsstr.preds;
acc3=sum(k3labels_ts==k3preds)./3.05;
conf3=confusionmat(uint8(k3preds)+1,uint8(k3labels_ts)+1);


meanacc=mean([acc0,acc1,acc2,acc3])
stdacc=std([acc0,acc1,acc2,acc3])
confall=(conf0+conf1+conf2+conf3)

confusionchart(confall,labelnames)





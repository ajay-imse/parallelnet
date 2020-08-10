# -*- coding: utf-8 -*-

import torch, time, os

import torch.nn as nn
import torch.nn.functional as F

#from graphviz import Digraph
import scipy.io as sio
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
lens = 0.25

thresh = 0.3
decay = 0.3

#fileloc = '/home/neuro-intel-linux/Documents/yjwu_nmnist'

num_classes = 10
batch_size = 1
num_epochs = 20
learning_rate = 1e-4#########default 1e-4
time_window = 100
names = 'animales_stbp'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print (device)


# load datasets need (num,pol, size,size, step)

train_dataset=sio.loadmat('C:\\Users\\Ajay\\Documents\\IBMdataset\\DvsGesture\\ibmframes.mat')
tr=train_dataset['myframes']

tr_images = torch.from_numpy(tr)

print(tr_images.shape) #64,64,100,1219





train_labels=sio.loadmat('C:\\Users\\Ajay\\Documents\\IBMdataset\\DvsGesture\\ibmlabels.mat')
trl=train_labels['labels']

tr_labels = torch.from_numpy(trl)


class ActFun(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        #print('\nActFun forward called',input.size())
        return input.gt(thresh).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        temp = abs(input - thresh) < lens
        #print('start \n',temp.size(),'\n', grad_input.size(), torch.mean(grad_input), 'fin \n')
        #print('actfun backward called')
        return grad_input * temp.float() / (2 * lens)        
        #return grad_input * 1 / (2 * lens)
        




def lr_scheduler(optimizer, epoch, init_lr=0.1, lr_decay_epoch=100):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""

    if epoch % lr_decay_epoch == 0 and epoch > 1:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.1

    return optimizer


best_acc = 0
acc_record = list([])


class SNN_Model(nn.Module):

    def __init__(self, num_classes=10):
        super(SNN_Model, self).__init__()
        self.conv0 = nn.Conv2d(1,20, kernel_size=5, stride=1, padding=2)
        self.conv1 = nn.Conv2d(20, 50, kernel_size=5, stride=1, padding=2)

        #in_planes, out_planes, stride = cfg_cnn[2]
        #self.conv2 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1)

        self.fc1 = nn.Linear(16*16 * 50, 200)
        self.fc2 = nn.Linear(200, 10)

    def forward(self, input, win, bsize):
        #input to be bsize x 28 x 28
        c0_mem = c0_spike = torch.zeros( 20, 64, 64, device=device)
        p0_mem = p0_spike = torch.zeros( 20, 32, 32, device=device)

        c1_mem = c1_spike = torch.zeros( 50, 32, 32, device=device)
        p1_mem = p1_spike = torch.zeros( 50, 16, 16, device=device)

        #c2_mem = c2_spike = torch.zeros(bsize, cfg_cnn[2][1], 7, 7, device=device)
        #p2_mem = p2_spike = torch.zeros(bsize, cfg_cnn[2][1], 7, 7, device=device)

        h1_mem = h1_spike = h1_sumspike = torch.zeros( 200, device=device)
        h2_mem = h2_spike = h2_sumspike = torch.zeros( 10, device=device)

        x=torch.zeros(input.size()).float()

        #firstspikesidx=torch.zeros(batch_size,5)
        #spkcnt=torch.zeros(batch_size).byte()
        #input = input.float()
        for step in range(win):
            #x = input[:, :, :, :, step]
            x = input[:, :, step]
            x=x[:,:,None,None]
            x=x.permute(2,3,0,1)

            c0_mem, c0_spike = mem_update(self.conv0, x, c0_mem, c0_spike,decay,step)
            p0_mem, p0_spike = mem_update_pool(F.avg_pool2d, c0_spike, p0_mem, p0_spike,decay,step)

            c1_mem, c1_spike = mem_update(self.conv1, p0_spike, c1_mem, c1_spike,decay,step)
            p1_mem, p1_spike = mem_update_pool(F.avg_pool2d, c1_spike, p1_mem, p1_spike,decay,step)

            #c2_mem, c2_spike = mem_update(self.conv2, p1_spike, c2_mem, c2_spike,decay,step)
            #p2_mem, p2_spike = mem_update_pool(F.avg_pool2d, c2_spike, p2_mem, p2_spike,decay,step)

            #x = c2_spike.view(bsize, -1)
            x = p1_spike.view( -1)

            h1_mem, h1_spike = mem_update(self.fc1, x, h1_mem, h1_spike,decay,step)
            h1_sumspike += h1_spike

            #h2_mem = mem_update_last(self.fc2, h1_spike, h2_mem,decay,step)
            h2_mem, h2_spike = mem_update(self.fc2, h1_spike, h2_mem, h2_spike,decay,step)
            h2_sumspike += h2_spike
  
        outputs = h2_sumspike / time_window
        
        return outputs




def mem_update(conv, x, mem, spike,curr_decay,curr_step):
    #mem = mem * curr_decay *float(time_window-curr_step)/float(time_window)* (1 - spike) + conv(x)
    mem = mem * (1 - spike) *curr_decay + conv(x)
    spike = act_fun(mem)
    return mem, spike

def mem_update_last(conv, x, mem,curr_decay,curr_step):
    mem = mem *curr_decay + conv(x)
    return mem




def mem_update_pool(opts, x, mem, spike,curr_decay,curr_step):
    mem = mem * float(time_window-curr_step)/float(time_window)*curr_decay*(1 - spike) + opts(x, 2)
    #mem = mem * (1 - spike) *curr_decay + opts(x, 2)
    spike = act_fun(mem)
    return mem, spike


snn = SNN_Model()
print(torch.sum(snn.conv0.weight.data))
print(torch.max(snn.conv0.weight.data))

snn.to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(snn.parameters(), lr=learning_rate)

act_fun = ActFun.apply

tr_images=tr_images.float()
ts_images = tr_images.float()
tr_labels=tr_labels.long()

preds=np.zeros_like(tr_labels)
bestresult=0.0
for epoch in range(num_epochs):

    running_loss = 0
    start_time = time.time()

    numiters = 1219#60000//batch_size
    shortiter=numiters

    for i in range(shortiter):
        #images = tr_images[i,:,:,:,:,:]
        images = tr_images[:,:,:,i]
        #images = images[:,:,:,None,None]
        #images.permute(2,3,0,1)




        #print('\n','iteration num',i,'\n')
        snn.zero_grad()
        optimizer.zero_grad()
        #print(images[0,0,:,:])
        images = images.float().to(device)
        outputs = snn(images, time_window,1)
        ops = outputs.cpu()

        #print(outputs)
        lbls=torch.FloatTensor(10)
        lbls.zero_()
        lbls[tr_labels[i]]=1.
        loss = criterion(outputs.cpu(), lbls)

        running_loss += loss.item()
        loss.backward()
        optimizer.step()

        if (i + 1) % 50 == 0:
            #print (torch.sum(outputs,1))
            print('Epoch [%d/%d], Step [%d/%d], Loss: %.5f'
                  % (epoch + 1, num_epochs, i + 1, numiters, running_loss))
            running_loss = 0
            print('Time elasped:', time.time() - start_time)
    correct = 0
    total = 0

    optimizer = lr_scheduler(optimizer, epoch, learning_rate, 40)


    for i in range(shortiter):
        images = ts_images[:,:,:,i]
        
        
        lbls=torch.FloatTensor(10)
        lbls.zero_()
        lbls[tr_labels[i]]=1.

        images = images.float().to(device)
        outputs = snn(images, time_window,1) #/ time_window
        
        _,preds[i]=torch.max(outputs.data,0)

        #_, predicted = torch.max(outputs.data, 0)
        #_, lbls = torch.max(lbls.data, 0)
        #total += lbls.size(0)
        #correct += (predicted.cpu() == lbls).sum()
        #correct += (predicted.cpu() == tr_labels).sum()
        #print(correct,total)


        #i +=1
    ct = (preds[0:shortiter]==tr_labels[0:shortiter])
    #print(preds[0:shortiter])
    print('\n\n\n')
    #print(tr_labels[0:shortiter])
    #print('\n\n\n')
    #print(preds.shape,tr_labels.shape)
    #print(ct.shape)
    print('Iters:', epoch, '\n')
    correct=ct.sum()
    print(correct)
    print('Test Accuracy of the model: %.3f  ' % (100 * correct.float() / shortiter))
    
    if correct.float()>=bestresult:
        bestresult=correct.float()
        sio.savemat('bestresults_preds.mat',{'preds':preds})
        print('saved')
        
    
    print(torch.sum(snn.conv0.weight.data))
    print(torch.max(snn.conv0.weight.data))

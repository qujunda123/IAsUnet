from bdataloader import myDataset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import torch.optim as optim
from torch.optim import lr_scheduler


import time
from tqdm import tqdm
from visdom import Visdom
from tensorboardX import SummaryWriter
import torchvision.utils as vutils
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
SumWriter=SummaryWriter(log_dir="./log")
viz=Visdom()

def conv_block(inc,out):
    return nn.Sequential(
        nn.Conv3d(in_channels=inc,out_channels=out,kernel_size=3,padding=1),
        nn.GroupNorm(num_groups=8, num_channels=out, eps=1e-01, affine=True),
        nn.PReLU(),
        nn.Conv3d(in_channels=out,out_channels=out,kernel_size=3,padding=1),
        nn.GroupNorm(num_groups=8, num_channels=out, eps=1e-01, affine=True),
        nn.PReLU()
        )

def up_conv(inc,out):
    return nn.Sequential(
        nn.ConvTranspose3d(in_channels=inc,out_channels=out,kernel_size=2,stride=2),
        nn.GroupNorm(num_groups=8, num_channels=out, eps=1e-01, affine=True),
        nn.PReLU())
 
def featureScale(ins, out):
    return nn.Sequential(
        nn.Conv3d(ins, out, 1, bias=False),
        nn.GroupNorm(num_groups=8, num_channels=out, eps=1e-01, affine=True),
        nn.PReLU()
    )



class groupBlock(nn.Module):
    def __init__(self,inc,outc):
        super(groupBlock,self).__init__()
        d=[1,2,3,4]
        self.inChannels=inc//4
        self.outChannels=inc//4
        self.grouped=nn.ModuleList()
        for i in range(4):
            self.grouped.append(
                nn.Sequential(
                    nn.Conv3d(
                        self.inChannels,
                        self.outChannels,3,padding=d[i],dilation=d[i],bias=True),
                        nn.GroupNorm(num_groups=2, num_channels=self.outChannels, eps=1e-01, affine=True),
                        nn.PReLU()
                    )
                )
    
    def forward(self,x):
        x_split=torch.split(x,self.inChannels,dim=1)
        x=[conv(t) for conv,t in zip(self.grouped,x_split)]
        x=torch.cat(x,dim=1)
        return x

class Attention(nn.Module):
    """
    modified from https://github.com/wulalago/FetalCPSeg/Network/MixAttNet.py
    """
    def __init__(self,inc,outc):
        super(Attention,self).__init__()
        self.group=groupBlock(inc,outc)
        self.conv1=nn.Conv3d(outc,outc,kernel_size=1)
        self.group1=groupBlock(outc,outc)
        self.conv2=nn.Conv3d(outc,outc,kernel_size=1)
        self.norm=nn.GroupNorm(num_groups=8, num_channels=outc, eps=1e-01, affine=True)
        self.norm1=nn.GroupNorm(num_groups=8, num_channels=outc, eps=1e-01, affine=True)
        self.act=nn.PReLU()
    
    def forward(self,x):
        group1=self.conv1(self.group(x))
        group2=self.conv2(group1)
        att=F.sigmoid(self.conv2(group2))
        out=self.norm(x*att)+self.norm1(x)
        return self.act(out),att
    
class model(nn.Module):
    def __init__(self,number=32,phase='train'):
        super().__init__()
        self.phase=phase
        self.conv1=conv_block(1,number)
        self.conv2=conv_block(number+1,number*2)
        self.conv3=conv_block(number+1+number*2,number*4)
        self.conv4=conv_block(number+1+number*2+number*4,number*8)
        self.conv5=conv_block(number+1+number*2+number*4+number*8,number*16)
        self.up6=up_conv(number+1+number*2+number*4+number*8+number*16,number*8)
        self.conv6=conv_block(number*16,number*8)
        self.up7=up_conv(number*16+number*8,number*4)
        self.conv7=conv_block(number*8,number*4)
        self.up8=up_conv(number*8+number*4,number*2)
        self.conv8=conv_block(number*4,number*2)
        self.up9=up_conv(number*4+number*2,number)
        self.conv9=conv_block(number*2,number)
        self.maxpool=nn.MaxPool3d(2)
        self.featureScale1=featureScale(number*3,int(number/2))
        self.featureScale2=featureScale(number*6,int(number/2))
        self.featureScale3=featureScale(number*12,int(number/2))
        self.featureScale4=featureScale(number*24,int(number/2))
        self.conv1x1=nn.Conv3d(int(number/2),1, kernel_size=1)
        self.conv1x2=nn.Conv3d(int(number/2), 1, kernel_size=1)
        self.conv1x3=nn.Conv3d(int(number/2), 1, kernel_size=1)
        self.conv1x4=nn.Conv3d(int(number/2), 1, kernel_size=1)
        self.mix1=Attention(int(number/2),int(number/2))
        self.mix2=Attention(int(number/2),int(number/2))
        self.mix3=Attention(int(number/2),int(number/2))
        self.mix4=Attention(int(number/2),int(number/2))
        self.conv2x1=nn.Conv3d(int(number/2),1,kernel_size=1)
        self.conv2x2=nn.Conv3d(int(number/2),1,kernel_size=1)
        self.conv2x3=nn.Conv3d(int(number/2),1,kernel_size=1)
        self.conv2x4=nn.Conv3d(int(number/2),1,kernel_size=1)
        self.final=nn.Sequential(
            nn.Conv3d(number*2, number*2, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=number*2, eps=1e-01, affine=True),
            nn.PReLU(),
            nn.Conv3d(number*2, 1, kernel_size=1))
        

        for m in self.modules():
            if isinstance(m,(nn.Conv3d,nn.ConvTranspose3d,nn.Linear)):
                nn.init.xavier_normal_(m.weight, gain=1.0)
            elif isinstance(m,(nn.BatchNorm3d,nn.GroupNorm)):
                nn.init.constant_(m.weight,1)
                nn.init.constant_(m.bias,0)

    def forward(self,x):
        conv1=self.conv1(x)
        pool1=torch.cat([x,conv1],dim=1)
        pool1=self.maxpool(pool1)

        conv2=self.conv2(pool1)
        pool2=torch.cat([pool1,conv2],dim=1)
        pool2=self.maxpool(pool2)

        conv3=self.conv3(pool2)
        pool3=torch.cat([pool2,conv3],dim=1)
        pool3=self.maxpool(pool3)

        conv4=self.conv4(pool3)
        pool4=torch.cat([pool3,conv4],dim=1)
        pool4=self.maxpool(pool4)

        conv5=self.conv5(pool4)
        conv5=torch.cat([pool4,conv5],dim=1)

        up6=torch.cat([self.up6(conv5),conv4],dim=1)
        conv6=self.conv6(up6)
        conv6=torch.cat([up6,conv6],dim=1)

        up7=torch.cat([self.up7(conv6),conv3],dim=1)
        conv7=self.conv7(up7)       
        conv7=torch.cat([up7,conv7],dim=1)


        up8=torch.cat([self.up8(conv7),conv2],dim=1)
        conv8=self.conv8(up8)
        conv8=torch.cat([up8,conv8],dim=1)


        up9=torch.cat([self.up9(conv8),conv1],dim=1)
        conv9=self.conv9(up9)
        conv9=torch.cat([up9,conv9],dim=1)

        fs1=F.interpolate(self.featureScale1(conv9),x.size()[2:],mode='trilinear')
        fs2=F.interpolate(self.featureScale2(conv8),x.size()[2:],mode='trilinear')
        fs3=F.interpolate(self.featureScale3(conv7),x.size()[2:],mode='trilinear')
        fs4=F.interpolate(self.featureScale4(conv6),x.size()[2:],mode='trilinear')
        
        fs1o=self.conv1x1(fs1)
        fs2o=self.conv1x2(fs2)
        fs3o=self.conv1x3(fs3)
        fs4o=self.conv1x4(fs4)
        
        mix1,att1=self.mix1(fs1)
        mix2,att2=self.mix2(fs2)
        mix3,att3=self.mix3(fs3)
        mix4,att4=self.mix4(fs4)
        
        mix_out1=self.conv2x1(mix1)
        mix_out2=self.conv2x2(mix2)
        mix_out3=self.conv2x3(mix3)
        mix_out4=self.conv2x4(mix4)
        out=torch.cat((mix1,mix2,mix3,mix4),dim=1)
        out=self.final(out)
        if self.phase=='train':
            return {'out':out,'as1':mix_out1,'as2':mix_out2,'as3':mix_out3,'as4':mix_out4,'s1':fs1o,'s2':fs2o,'s3':fs3o,'s4':fs4o}
        else:
            return out

import numpy as np
def dice_coef(y_true, y_pred):
    smooth = 1e-5
    y_true=y_true.data.cpu().numpy()
    y_pred=F.sigmoid(y_pred).data.cpu().numpy()
    y_true_f = y_true.reshape(len(y_true),-1)
    y_pred_f = y_pred.reshape(len(y_true),-1)
    intersection = np.sum(y_true_f * y_pred_f,axis=1)

    dice=(2.*intersection + smooth) / (np.sum(y_true_f,axis=1) + np.sum(y_pred_f,axis=1) + smooth)
    return np.sum(dice)/(len(y_true))

class SoftDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(SoftDiceLoss, self).__init__()

    def forward(self, logits, targets):
        num = targets.size(0)
        smooth = 1e-5

        probs = F.sigmoid(logits)
        m1 = probs.view(num, -1)

        m2 = targets.contiguous().view(num, -1)
        intersection = (m1 * m2)

        score =  (2. *intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(1) + smooth)
        score = 1 - score.sum() / (num)

        return score

class LogCoshDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(LogCoshDiceLoss, self).__init__()

    def forward(self, logits, targets):
        num = targets.size(0)
        smooth = 1e-5

        probs = F.sigmoid(logits)
        m1 = probs.view(num, -1)

        m2 = targets.contiguous().view(num, -1)
        intersection = (m1 * m2)

        score =  (2. *intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(1) + smooth)
        score = 1 - score.sum() / (num)

        return torch.log((torch.exp(score)+torch.exp(-score))/2.0)
 
    
 
from others import AvgMeter
batch_size1=1
epoch=150
folder='./dataset/'
traindataset=myDataset(folder)
valdataset=myDataset(folder,'val')
trainloader=DataLoader(traindataset,batch_size=batch_size1,shuffle=True,pin_memory=True)
valloader=DataLoader(valdataset,batch_size=batch_size1,shuffle=False,pin_memory=True)


device='cuda'
model=model(phase='train',number=64).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4,weight_decay=0)
lrsch=lr_scheduler.ReduceLROnPlateau(optimizer,mode='max',factor=0.1,patience=30)

lossm=AvgMeter()

cost=SoftDiceLoss()

import torch.backends.cudnn as cudnn
cudnn.enabled = True
cudnn.benchmark = True
numi=0
numt=0
accumu=4
# model.load_state_dict(torch.load('./643003.pth.gz'))
for i in range(1,epoch+1):
    start=time.time()
    model.train()
    train_dice=0
    step=0
    for x,y in trainloader:
        x,y=x.float().to(device),y.to(device)
        pPixelW=(torch.FloatTensor([x.size(0)*x.size(2)*x.size(3)*x.size(4)])/torch.where(y==1)[0].size(0)).to(device)
        predict=model(x)
        torch.cuda.empty_cache()
        l1=cost(predict['out'], y)
        l2=0.8*cost(predict['as1'], y)
        l3=0.7*cost(predict['as2'], y)
        l4=0.6*cost(predict['as3'], y)
        l5=0.5*cost(predict['as4'], y)
        l6=0.8*cost(predict['s1'], y)
        l7=0.7*cost(predict['s2'], y)
        l8=0.6*cost(predict['s3'], y)
        l9=0.5*cost(predict['s4'], y)
        loss=l1+l2+l3+l4+l5+l6+l7+l8+l9
        dice=dice_coef(y,predict['out'])
        number=y.size(0)
        train_dice +=dice*number
        train_loss=loss.item()        
        loss=loss/accumu
        loss.backward()
        if (step+1)%accumu==0:
            optimizer.step()
            optimizer.zero_grad()
        lossm.update(train_loss)
        

        numi+=1
        step+=1
        if step % 50==0:
            print('[{}|{}]--loss:{:.6f},Dice:{:.6f}'.format(i,numi,train_loss,dice))       
            viz.line([[train_loss],[dice]],[numi],win='train_loss',opts=dict(title='train_loss_dice',legend=['loss', 'acc']),update="append")
    torch.save(model.state_dict(), "./multiscale.pth.gz", _use_new_zipfile_serialization=False)
    print('Train loss:{:.6f},trainDice:{:.6f},LR:{:.6f},Time:{:.6f}'.format(lossm.avg,train_dice/((len(trainloader)-1)*batch_size1+number),optimizer.param_groups[0]['lr'],time.time()-start))
    lossm.reset()
    
    model.eval()
    val_dice=0
    val_num=0
    for x,y in valloader:
        x,y=x.float().to(device),y.long().to(device)
        number=y.size(0)
        with torch.no_grad():
            predict=model(x)
        
        dice=dice_coef(y,predict['out'])
        val_dice +=dice*number
        val_num+=1
    print('valDice:{:.6f}'.format(val_dice/((len(valloader)-1)*batch_size1+number)))
    lrsch.step(val_dice/((len(valloader)-1)*batch_size1+number))
    numt +=1
    viz.line([[val_dice/((len(valloader)-1)*batch_size1+number)]],[numt],win='test',opts=dict(title='test',legend=['acc']),update='append')


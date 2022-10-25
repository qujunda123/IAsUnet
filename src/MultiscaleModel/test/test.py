#%%
import warnings

warnings.filterwarnings('ignore')
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
from skimage import measure

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
 
# class model(nn.Module):
#     def __init__(self,number=32,phase='train'):
#         super().__init__()
#         self.phase=phase
#         self.dropout=nn.FeatureAlphaDropout(p=0.1,inplace=True)
#         self.conv1=conv_block(1,number)
#         self.conv2=conv_block(number+1,number*2)
#         self.conv3=conv_block(number+1+number*2,number*4)
#         self.conv4=conv_block(number+1+number*2+number*4,number*8)
#         self.conv5=conv_block(number+1+number*2+number*4+number*8,number*16)
#         self.up6=up_conv(number+1+number*2+number*4+number*8+number*16,number*8)
#         self.conv6=conv_block(number*16,number*8)
#         self.up7=up_conv(number*16+number*8,number*4)
#         self.conv7=conv_block(number*8,number*4)
#         self.up8=up_conv(number*8+number*4,number*2)
#         self.conv8=conv_block(number*4,number*2)
#         self.up9=up_conv(number*4+number*2,number)
#         self.conv9=conv_block(number*2,number)
#         self.maxpool=nn.MaxPool3d(2)
#         self.featureScale1=featureScale(number*3,int(number/2))
#         self.featureScale2=featureScale(number*6,int(number/2))
#         self.featureScale3=featureScale(number*12,int(number/2))
#         self.featureScale4=featureScale(number*24,int(number/2))
#         self.conv1x1=nn.Conv3d(int(number/2),1, kernel_size=1)
#         self.conv1x2=nn.Conv3d(int(number/2), 1, kernel_size=1)
#         self.conv1x3=nn.Conv3d(int(number/2), 1, kernel_size=1)
#         self.conv1x4=nn.Conv3d(int(number/2), 1, kernel_size=1)
#         self.mix1=Attention(int(number/2),int(number/2))
#         self.mix2=Attention(int(number/2),int(number/2))
#         self.mix3=Attention(int(number/2),int(number/2))
#         self.mix4=Attention(int(number/2),int(number/2))
#         self.conv2x1=nn.Conv3d(int(number/2),1,kernel_size=1)
#         self.conv2x2=nn.Conv3d(int(number/2),1,kernel_size=1)
#         self.conv2x3=nn.Conv3d(int(number/2),1,kernel_size=1)
#         self.conv2x4=nn.Conv3d(int(number/2),1,kernel_size=1)
#         self.final=nn.Sequential(
#             nn.Conv3d(number*2, number*2, kernel_size=3, padding=1),
#             nn.GroupNorm(num_groups=8, num_channels=number*2, eps=1e-01, affine=True),
#             nn.PReLU(),
#             nn.Conv3d(number*2, 1, kernel_size=1))
        

#         for m in self.modules():
#             if isinstance(m,(nn.Conv3d,nn.ConvTranspose3d,nn.Linear)):
#                 nn.init.xavier_normal_(m.weight, gain=1.0)
#             elif isinstance(m,(nn.BatchNorm3d,nn.GroupNorm)):
#                 nn.init.constant_(m.weight,1)
#                 nn.init.constant_(m.bias,0)

#     def forward(self,x):
#         conv1=self.conv1(x)
#         pool1=self.dropout(torch.cat([x,conv1],dim=1))
#         pool1=self.maxpool(pool1)

#         conv2=self.conv2(pool1)
#         pool2=self.dropout(torch.cat([pool1,conv2],dim=1))
#         pool2=self.maxpool(pool2)

#         conv3=self.conv3(pool2)
#         pool3=self.dropout(torch.cat([pool2,conv3],dim=1))
#         pool3=self.maxpool(pool3)

#         conv4=self.conv4(pool3)
#         pool4=self.dropout(torch.cat([pool3,conv4],dim=1))
#         pool4=self.maxpool(pool4)

#         conv5=self.conv5(pool4)
#         conv5=self.dropout(torch.cat([pool4,conv5],dim=1))

#         up6=self.dropout(torch.cat([self.up6(conv5),conv4],dim=1))
#         conv6=self.conv6(up6)
#         conv6=self.dropout(torch.cat([up6,conv6],dim=1))

#         up7=self.dropout(torch.cat([self.up7(conv6),conv3],dim=1))
#         conv7=self.conv7(up7)       
#         conv7=self.dropout(torch.cat([up7,conv7],dim=1))


#         up8=self.dropout(torch.cat([self.up8(conv7),conv2],dim=1))
#         conv8=self.conv8(up8)
#         conv8=self.dropout(torch.cat([up8,conv8],dim=1))


#         up9=self.dropout(torch.cat([self.up9(conv8),conv1],dim=1))
#         conv9=self.conv9(up9)
#         conv9=self.dropout(torch.cat([up9,conv9],dim=1))

#         fs1=F.interpolate(self.featureScale1(conv9),x.size()[2:],mode='trilinear')
#         fs2=F.interpolate(self.featureScale2(conv8),x.size()[2:],mode='trilinear')
#         fs3=F.interpolate(self.featureScale3(conv7),x.size()[2:],mode='trilinear')
#         fs4=F.interpolate(self.featureScale4(conv6),x.size()[2:],mode='trilinear')
        
#         fs1o=self.conv1x1(fs1)
#         fs2o=self.conv1x2(fs2)
#         fs3o=self.conv1x3(fs3)
#         fs4o=self.conv1x4(fs4)
        
#         mix1,att1=self.mix1(fs1)
#         mix2,att2=self.mix2(fs2)
#         mix3,att3=self.mix3(fs3)
#         mix4,att4=self.mix4(fs4)
        
#         mix_out1=self.conv2x1(mix1)
#         mix_out2=self.conv2x2(mix2)
#         mix_out3=self.conv2x3(mix3)
#         mix_out4=self.conv2x4(mix4)
#         out=torch.cat((mix1,mix2,mix3,mix4),dim=1)
#         out=self.final(out)
#         if self.phase=='train':
#             return {'out':out,'as1':mix_out1,'as2':mix_out2,'as3':mix_out3,'as4':mix_out4,'s1':fs1o,'s2':fs2o,'s3':fs3o,'s4':fs4o}
#         else:
#             return out    
 
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
    y_true_f = y_true.reshape(len(y_true),-1)
    y_pred_f = y_pred.reshape(len(y_true),-1)
    intersection = np.sum(y_true_f * y_pred_f,axis=1)
    dice=(2.*intersection + smooth) / (np.sum(y_true_f,axis=1) + np.sum(y_pred_f,axis=1) + smooth)
    return np.sum(dice)/(len(y_true))

device='cpu'
if device=='cuda':
    half='half'
else:
    half='float'
if half=='half':
    model=model().half().to(device)
else:
    model=model(number=64).to(device)
# model.load_state_dict(torch.load('../model/o150le-2.pkl'))
TTA=False
model.load_state_dict(torch.load('../644502.pth.gz'))
# model.load_state_dict(torch.load('../32fu300two.pth.gz')['model'])
import glob
import SimpleITK as sitk
file=glob.glob('../../qinghua/test/data/*')
dicep=[]
SENS=[]
SPEC=[]
PREC=[]
BACC=[]
AUC=[]
KAP=[]
VS=[]
NMCC=[]
AHD=[] 

meana=[]
stda=[]
model.eval()

from miseval import evaluate
if TTA:
    import ttach as tta
    transforms=tta.Compose([
            #tta.HorizontalFlip(),
            tta.VerticalFlip(),
            # tta.Rotate90(angles=[90,180,270])
            # tta.Resize(sizes=[480]),       
        ])
    tta_model=tta.SegmentationTTAWrapper(model, transforms,merge_mode='tsharpen')
    tta_model.eval()

import pickle
f=open('../test/testi.pkl','rb')
testi=pickle.load(f)
f.close()
f=open('../test/testm.pkl','rb')
testm=pickle.load(f)
f.close()

for zz in tqdm(range(len(file))):
# for zz in tqdm(range(1,2)):
    image=[]
    mask=[]
    imagefile=[]
    maskfile=[]
    mfile=glob.glob(file[zz]+'/*.nii.gz')
    if len(mfile)==0:
        mfile=glob.glob(file[zz]+'/*.nii')
    maskfile.append(mfile[0])
        
    names=glob.glob(file[zz]+'/*.dcm')
    f=file[zz].replace('data','brain_data')
    iii=sitk.ReadImage(f+'/image.nii.gz') 
    m=sitk.ReadImage(maskfile[0])
    ii=testi[zz]
    mm=testm[zz]

    if half=='half':
        ii=torch.as_tensor(ii).type(torch.float16).to(device)
    else:
        ii=torch.as_tensor(ii).float().to(device)
    
    if TTA:
        with torch.no_grad():
            if ii.size(2)!=128:
                pred=model(ii)
    
            else:
                pred=tta_model(ii)
        if half=='half':
            imgs_mask_test = pred.cpu().numpy()
    
        else:
            if type(pred)==dict:
                imgs_mask_test = pred['out'].numpy()
            else:
                imgs_mask_test = pred.numpy()
        
    else:
        with torch.no_grad():
            pred=model(ii)       
            pred=pred['out'] 
        if half=='half':
            imgs_mask_test = F.sigmoid(pred).cpu().numpy()
    
        else:
            imgs_mask_test = F.sigmoid(pred).numpy()
    # 
    np.save('./result/64fu/'+str(zz)+'.npy',imgs_mask_test)
    
    imgs_mask_test[imgs_mask_test>=0.5]=1
    imgs_mask_test[imgs_mask_test<0.5]=0
    imgs_mask_test=imgs_mask_test
    labels1=measure.label(imgs_mask_test[0,0,:,:,:])
    region1=measure.regionprops(labels1)
    unitVol=np.prod(m.GetSpacing())
    for qq in range(len(region1)):
        bbox=region1[qq].bbox
        counts=np.sum(imgs_mask_test[0,0,bbox[0]:bbox[-3],bbox[1]:bbox[-2],bbox[2]:bbox[-1]])
        if counts*unitVol<=5:
            imgs_mask_test[0,0,bbox[0]:bbox[-3],bbox[1]:bbox[-2],bbox[2]:bbox[-1]]=0
    dice=dice_coef(imgs_mask_test,mm)
    mm=mm[0,0]
    sens=evaluate(mm,imgs_mask_test[0,0],metric='SENS')
    spec=evaluate(mm,imgs_mask_test[0,0],metric='SPEC')
    prec=evaluate(mm,imgs_mask_test[0,0],metric='PREC')
    bacc=evaluate(mm,imgs_mask_test[0,0],metric='BACC')
    auc=evaluate(mm,imgs_mask_test[0,0],metric='AUC')
    kap=evaluate(mm,imgs_mask_test[0,0],metric='KAP')
    vs=evaluate(mm,imgs_mask_test[0,0],metric='VS')
    nmcc=evaluate(mm,imgs_mask_test[0,0],metric='nMCC')
    ahd=0
    for iii in range(imgs_mask_test.shape[2]):
        ahd +=evaluate(mm[iii],imgs_mask_test[0,0,iii],metric='AHD')
    ahd=ahd/imgs_mask_test.shape[2]
    print('Filename:{}\nDice:{}'.format(file[zz],dice))
    dicep.append(dice)
    SENS.append(sens)
    SPEC.append(spec)
    PREC.append(prec)
    BACC.append(bacc)
    AUC.append(auc)
    KAP.append(kap)
    VS.append(vs)
    NMCC.append(nmcc)
    AHD.append(ahd)
print('\n\n\nAveragedDice:{}\nSensitivity:{}\nSpecificity:{}\nPrecision:{}\n\
Banlanced ACC:{}\nAUC:{}\nKAP:{}\nVolumetric Similarity:{}\nNormalized Matthews Correlation Coefficient:{}\n\
Averaged Hf:{}\n'.format(np.mean(dicep),np.mean(SENS),np.mean(SPEC),np.mean(PREC),np.mean(BACC),np.mean(AUC),np.mean(KAP),\
              np.mean(VS),np.mean(NMCC),np.mean(AHD))) 
#%%
import warnings

warnings.filterwarnings('ignore')
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
from skimage import measure

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
    y_true_f = y_true.reshape(len(y_true),-1)
    y_pred_f = y_pred.reshape(len(y_true),-1)
    intersection = np.sum(y_true_f * y_pred_f,axis=1)

    dice=(2.*intersection + smooth) / (np.sum(y_true_f,axis=1) + np.sum(y_pred_f,axis=1) + smooth)
    return np.sum(dice)/(len(y_true))

device='cpu'
if device=='cuda':
    half='half'
else:
    half='float'
if half=='half':
    model=model().half().to(device)
else:
    model=model(number=32).to(device)
# model.load_state_dict(torch.load('../jclahe300.pth.gz'))
model.load_state_dict(torch.load('../qclahe150.pth.gz')['model'])
import glob
import SimpleITK as sitk
file=glob.glob('../../qinghua/test/data/*')
dicep=[]
SENS=[]
SPEC=[]
PREC=[]
BACC=[]
AUC=[]
KAP=[]
VS=[]
NMCC=[]
AHD=[]

meana=[]
stda=[]
model.eval()
TTA=False
from miseval import evaluate
if TTA:
    import ttach as tta
    transforms=tta.Compose([
            #tta.HorizontalFlip(),
            tta.VerticalFlip(),
            # tta.Rotate90(angles=[90,180,270])
            # tta.Resize(sizes=[480]),       
        ])
    tta_model=tta.SegmentationTTAWrapper(model, transforms,merge_mode='tsharpen')
    tta_model.eval()

import pickle
import mclahe as mc
f=open('../test/testi.pkl','rb')
testi=pickle.load(f)
f.close()
f=open('../test/testm.pkl','rb')
testm=pickle.load(f)
f.close()

for zz in tqdm(range(len(file))):
# for zz in tqdm(range(1,2)):
    image=[]
    mask=[]
    imagefile=[]
    maskfile=[]
    mfile=glob.glob(file[zz]+'/*.nii.gz')
    if len(mfile)==0:
        mfile=glob.glob(file[zz]+'/*.nii')
    maskfile.append(mfile[0])
        
    names=glob.glob(file[zz]+'/*.dcm')
    f=file[zz].replace('data','brain_data')
    iii=sitk.ReadImage(f+'/image.nii.gz') 
    ii=sitk.GetArrayFromImage(iii)
    ii=np.clip(ii,0,2000)
    m=sitk.ReadImage(maskfile[0])
    mm=sitk.GetArrayFromImage(m)
    
   
    if ii.shape[0]==mm.shape[0]:
        pass
    else:
        ii=ii[0:96]
        mm=mm[0:96]
    
    sixteen=len(ii)/16
    six=np.round(sixteen)
    if six>sixteen:
        slice=int(six*16)
        var=slice-len(ii)
        i2=np.zeros((slice,384,384))
        m2=np.zeros((slice,384,384))
        var2=var//2
        i2[var2:len(ii)+var2]=ii[:,106:490,125:509]
        m2[var2:len(ii)+var2]=mm[:,106:490,125:509]
    elif six<sixteen:
        slice=int(six*16)
        var=len(ii)-slice
        var2=var//2
        i2=ii[var2:slice+var2,106:490,125:509]
        m2=mm[var2:slice+var2,106:490,125:509]
    else:
        i2=ii[:,106:490,125:509]
        m2=mm[:,106:490,125:509]
                 
    img_rows = i2.shape[1]
    img_cols = i2.shape[2]
    img_depth = i2.shape[0]
    
    i2=mc.mclahe(i2,kernel_size=[16,84,84],n_bins=256,clip_limit=0.01)
    
    ii=i2.reshape((1,1,img_depth,img_rows,img_cols))
    mm=m2.reshape((1,1,img_depth,img_rows,img_cols)).astype(np.float64)
    
    min=np.mean(ii)
    max=np.std(ii)
    ii=(ii-min)/(max)
    
    if half=='half':
        ii=torch.as_tensor(ii).type(torch.float16).to(device)
    else:
        ii=torch.as_tensor(ii).float().to(device)
    
    if TTA:
        with torch.no_grad():
            if ii.size(2)!=128:
                pred=model(ii)
    
            else:
                pred=tta_model(ii)
        if half=='half':
            imgs_mask_test = pred.cpu().numpy()
    
        else:
            if type(pred)==dict:
                imgs_mask_test = pred['out'].numpy()
            else:
                imgs_mask_test = pred.numpy()
        
    else:
        with torch.no_grad():
            pred=model(ii)       
            pred=pred['out'] 
        if half=='half':
            imgs_mask_test = F.sigmoid(pred).cpu().numpy()
    
        else:
            imgs_mask_test = F.sigmoid(pred).numpy()

    imgs_mask_test[imgs_mask_test>=0.5]=1
    imgs_mask_test[imgs_mask_test<0.5]=0
    imgs_mask_test=imgs_mask_test
    labels1=measure.label(imgs_mask_test[0,0,:,:,:])
    region1=measure.regionprops(labels1)
    unitVol=np.prod(m.GetSpacing())
    for qq in range(len(region1)):
        bbox=region1[qq].bbox
        counts=np.sum(imgs_mask_test[0,0,bbox[0]:bbox[-3],bbox[1]:bbox[-2],bbox[2]:bbox[-1]])
        if counts*unitVol<=5:
            imgs_mask_test[0,0,bbox[0]:bbox[-3],bbox[1]:bbox[-2],bbox[2]:bbox[-1]]=0
    np.save('./result/clahefu/'+str(zz)+'.npy',imgs_mask_test[0,0])
    # np.save('./result/mask/'+str(zz)+'.npy',mm)
    
    dice=dice_coef(imgs_mask_test,mm)
    sens=evaluate(mm,imgs_mask_test[0,0],metric='SENS')
    spec=evaluate(mm,imgs_mask_test[0,0],metric='SPEC')
    prec=evaluate(mm,imgs_mask_test[0,0],metric='PREC')
    bacc=evaluate(mm,imgs_mask_test[0,0],metric='BACC')
    auc=evaluate(mm,imgs_mask_test[0,0],metric='AUC')
    kap=evaluate(mm,imgs_mask_test[0,0],metric='KAP')
    vs=evaluate(mm,imgs_mask_test[0,0],metric='VS')
    nmcc=evaluate(mm,imgs_mask_test[0,0],metric='nMCC')
    ahd=0
    for iii in range(imgs_mask_test.shape[2]):
        ahd +=evaluate(mm[0,0,iii],imgs_mask_test[0,0,iii],metric='AHD')
    ahd=ahd/imgs_mask_test.shape[2]
    print('Filename:{}\nDice:{}'.format(file[zz],dice))
    dicep.append(dice)
    SENS.append(sens)
    SPEC.append(spec)
    PREC.append(prec)
    BACC.append(bacc)
    AUC.append(auc)
    KAP.append(kap)
    VS.append(vs)
    NMCC.append(nmcc)
    AHD.append(ahd)
print('\n\n\nAveragedDice:{}\nSensitivity:{}\nSpecificity:{}\nPrecision:{}\n\
Banlanced ACC:{}\nAUC:{}\nKAP:{}\nVolumetric Similarity:{}\nNormalized Matthews Correlation Coefficient:{}\n\
Averaged Hf:{}\n'.format(np.mean(dicep),np.mean(SENS),np.mean(SPEC),np.mean(PREC),np.mean(BACC),np.mean(AUC),np.mean(KAP),\
              np.mean(VS),np.mean(NMCC),np.mean(AHD))) 
#%%
'''
TTA with resize 2 times bigger
'''
# import warnings

# warnings.filterwarnings('ignore')
# from torch.utils.data import DataLoader
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torchsummary import summary
# import torch.optim as optim
# from torch.optim import lr_scheduler

# import time
# from tqdm import tqdm
# from visdom import Visdom
# from tensorboardX import SummaryWriter
# import torchvision.utils as vutils
# import os
# from skimage import measure

# os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
# SumWriter=SummaryWriter(log_dir="./log")
# viz=Visdom()

# def conv_block(inc,out):
#     return nn.Sequential(
#         nn.Conv3d(in_channels=inc,out_channels=out,kernel_size=3,padding=1),
#         nn.GroupNorm(num_groups=8, num_channels=out, eps=1e-01, affine=True),
#         nn.PReLU(),
#         nn.Conv3d(in_channels=out,out_channels=out,kernel_size=3,padding=1),
#         nn.GroupNorm(num_groups=8, num_channels=out, eps=1e-01, affine=True),
#         nn.PReLU()
#         )

# def up_conv(inc,out):
#     return nn.Sequential(
#         nn.ConvTranspose3d(in_channels=inc,out_channels=out,kernel_size=2,stride=2),
#         nn.GroupNorm(num_groups=8, num_channels=out, eps=1e-01, affine=True),
#         nn.PReLU())
 
# def featureScale(ins, out):
#     return nn.Sequential(
#         nn.Conv3d(ins, out, 1, bias=False),
#         nn.GroupNorm(num_groups=8, num_channels=out, eps=1e-01, affine=True),
#         nn.PReLU()
#     )


# class groupBlock(nn.Module):
#     def __init__(self,inc,outc):
#         super(groupBlock,self).__init__()
#         d=[1,2,3,4]
#         self.inChannels=inc//4
#         self.outChannels=inc//4
#         self.grouped=nn.ModuleList()
#         for i in range(4):
#             self.grouped.append(
#                 nn.Sequential(
#                     nn.Conv3d(
#                         self.inChannels,
#                         self.outChannels,3,padding=d[i],dilation=d[i],bias=True),
#                         nn.GroupNorm(num_groups=2, num_channels=self.outChannels, eps=1e-01, affine=True),
#                         nn.PReLU()
#                     )
#                 )
    
#     def forward(self,x):
#         x_split=torch.split(x,self.inChannels,dim=1)
#         x=[conv(t) for conv,t in zip(self.grouped,x_split)]
#         x=torch.cat(x,dim=1)
#         return x

# class Attention(nn.Module):
#     """
#     modified from https://github.com/wulalago/FetalCPSeg/Network/MixAttNet.py
#     """
#     def __init__(self,inc,outc):
#         super(Attention,self).__init__()
#         self.group=groupBlock(inc,outc)
#         self.conv1=nn.Conv3d(outc,outc,kernel_size=1)
#         self.group1=groupBlock(outc,outc)
#         self.conv2=nn.Conv3d(outc,outc,kernel_size=1)
#         self.norm=nn.GroupNorm(num_groups=8, num_channels=outc, eps=1e-01, affine=True)
#         self.norm1=nn.GroupNorm(num_groups=8, num_channels=outc, eps=1e-01, affine=True)
#         self.act=nn.PReLU()
    
#     def forward(self,x):
#         group1=self.conv1(self.group(x))
#         group2=self.conv2(group1)
#         att=F.sigmoid(self.conv2(group2))
#         out=self.norm(x*att)+self.norm1(x)
#         return self.act(out),att
 
# # class model(nn.Module):
# #     def __init__(self,number=32,phase='train'):
# #         super().__init__()
# #         self.phase=phase
# #         self.dropout=nn.FeatureAlphaDropout(p=0.1,inplace=True)
# #         self.conv1=conv_block(1,number)
# #         self.conv2=conv_block(number+1,number*2)
# #         self.conv3=conv_block(number+1+number*2,number*4)
# #         self.conv4=conv_block(number+1+number*2+number*4,number*8)
# #         self.conv5=conv_block(number+1+number*2+number*4+number*8,number*16)
# #         self.up6=up_conv(number+1+number*2+number*4+number*8+number*16,number*8)
# #         self.conv6=conv_block(number*16,number*8)
# #         self.up7=up_conv(number*16+number*8,number*4)
# #         self.conv7=conv_block(number*8,number*4)
# #         self.up8=up_conv(number*8+number*4,number*2)
# #         self.conv8=conv_block(number*4,number*2)
# #         self.up9=up_conv(number*4+number*2,number)
# #         self.conv9=conv_block(number*2,number)
# #         self.maxpool=nn.MaxPool3d(2)
# #         self.featureScale1=featureScale(number*3,int(number/2))
# #         self.featureScale2=featureScale(number*6,int(number/2))
# #         self.featureScale3=featureScale(number*12,int(number/2))
# #         self.featureScale4=featureScale(number*24,int(number/2))
# #         self.conv1x1=nn.Conv3d(int(number/2),1, kernel_size=1)
# #         self.conv1x2=nn.Conv3d(int(number/2), 1, kernel_size=1)
# #         self.conv1x3=nn.Conv3d(int(number/2), 1, kernel_size=1)
# #         self.conv1x4=nn.Conv3d(int(number/2), 1, kernel_size=1)
# #         self.mix1=Attention(int(number/2),int(number/2))
# #         self.mix2=Attention(int(number/2),int(number/2))
# #         self.mix3=Attention(int(number/2),int(number/2))
# #         self.mix4=Attention(int(number/2),int(number/2))
# #         self.conv2x1=nn.Conv3d(int(number/2),1,kernel_size=1)
# #         self.conv2x2=nn.Conv3d(int(number/2),1,kernel_size=1)
# #         self.conv2x3=nn.Conv3d(int(number/2),1,kernel_size=1)
# #         self.conv2x4=nn.Conv3d(int(number/2),1,kernel_size=1)
# #         self.final=nn.Sequential(
# #             nn.Conv3d(number*2, number*2, kernel_size=3, padding=1),
# #             nn.GroupNorm(num_groups=8, num_channels=number*2, eps=1e-01, affine=True),
# #             nn.PReLU(),
# #             nn.Conv3d(number*2, 1, kernel_size=1))
        

# #         for m in self.modules():
# #             if isinstance(m,(nn.Conv3d,nn.ConvTranspose3d,nn.Linear)):
# #                 nn.init.xavier_normal_(m.weight, gain=1.0)
# #             elif isinstance(m,(nn.BatchNorm3d,nn.GroupNorm)):
# #                 nn.init.constant_(m.weight,1)
# #                 nn.init.constant_(m.bias,0)

# #     def forward(self,x):
# #         conv1=self.conv1(x)
# #         pool1=self.dropout(torch.cat([x,conv1],dim=1))
# #         pool1=self.maxpool(pool1)

# #         conv2=self.conv2(pool1)
# #         pool2=self.dropout(torch.cat([pool1,conv2],dim=1))
# #         pool2=self.maxpool(pool2)

# #         conv3=self.conv3(pool2)
# #         pool3=self.dropout(torch.cat([pool2,conv3],dim=1))
# #         pool3=self.maxpool(pool3)

# #         conv4=self.conv4(pool3)
# #         pool4=self.dropout(torch.cat([pool3,conv4],dim=1))
# #         pool4=self.maxpool(pool4)

# #         conv5=self.conv5(pool4)
# #         conv5=self.dropout(torch.cat([pool4,conv5],dim=1))

# #         up6=self.dropout(torch.cat([self.up6(conv5),conv4],dim=1))
# #         conv6=self.conv6(up6)
# #         conv6=self.dropout(torch.cat([up6,conv6],dim=1))

# #         up7=self.dropout(torch.cat([self.up7(conv6),conv3],dim=1))
# #         conv7=self.conv7(up7)       
# #         conv7=self.dropout(torch.cat([up7,conv7],dim=1))


# #         up8=self.dropout(torch.cat([self.up8(conv7),conv2],dim=1))
# #         conv8=self.conv8(up8)
# #         conv8=self.dropout(torch.cat([up8,conv8],dim=1))


# #         up9=self.dropout(torch.cat([self.up9(conv8),conv1],dim=1))
# #         conv9=self.conv9(up9)
# #         conv9=self.dropout(torch.cat([up9,conv9],dim=1))

# #         fs1=F.interpolate(self.featureScale1(conv9),x.size()[2:],mode='trilinear')
# #         fs2=F.interpolate(self.featureScale2(conv8),x.size()[2:],mode='trilinear')
# #         fs3=F.interpolate(self.featureScale3(conv7),x.size()[2:],mode='trilinear')
# #         fs4=F.interpolate(self.featureScale4(conv6),x.size()[2:],mode='trilinear')
        
# #         fs1o=self.conv1x1(fs1)
# #         fs2o=self.conv1x2(fs2)
# #         fs3o=self.conv1x3(fs3)
# #         fs4o=self.conv1x4(fs4)
        
# #         mix1,att1=self.mix1(fs1)
# #         mix2,att2=self.mix2(fs2)
# #         mix3,att3=self.mix3(fs3)
# #         mix4,att4=self.mix4(fs4)
        
# #         mix_out1=self.conv2x1(mix1)
# #         mix_out2=self.conv2x2(mix2)
# #         mix_out3=self.conv2x3(mix3)
# #         mix_out4=self.conv2x4(mix4)
# #         out=torch.cat((mix1,mix2,mix3,mix4),dim=1)
# #         out=self.final(out)
# #         if self.phase=='train':
# #             return {'out':out,'as1':mix_out1,'as2':mix_out2,'as3':mix_out3,'as4':mix_out4,'s1':fs1o,'s2':fs2o,'s3':fs3o,'s4':fs4o}
# #         else:
# #             return out    
 
# class model(nn.Module):
#     def __init__(self,number=32,phase='train'):
#         super().__init__()
#         self.phase=phase
#         self.conv1=conv_block(1,number)
#         self.conv2=conv_block(number+1,number*2)
#         self.conv3=conv_block(number+1+number*2,number*4)
#         self.conv4=conv_block(number+1+number*2+number*4,number*8)
#         self.conv5=conv_block(number+1+number*2+number*4+number*8,number*16)
#         self.up6=up_conv(number+1+number*2+number*4+number*8+number*16,number*8)
#         self.conv6=conv_block(number*16,number*8)
#         self.up7=up_conv(number*16+number*8,number*4)
#         self.conv7=conv_block(number*8,number*4)
#         self.up8=up_conv(number*8+number*4,number*2)
#         self.conv8=conv_block(number*4,number*2)
#         self.up9=up_conv(number*4+number*2,number)
#         self.conv9=conv_block(number*2,number)
#         self.maxpool=nn.MaxPool3d(2)
#         self.featureScale1=featureScale(number*3,int(number/2))
#         self.featureScale2=featureScale(number*6,int(number/2))
#         self.featureScale3=featureScale(number*12,int(number/2))
#         self.featureScale4=featureScale(number*24,int(number/2))
#         self.conv1x1=nn.Conv3d(int(number/2),1, kernel_size=1)
#         self.conv1x2=nn.Conv3d(int(number/2), 1, kernel_size=1)
#         self.conv1x3=nn.Conv3d(int(number/2), 1, kernel_size=1)
#         self.conv1x4=nn.Conv3d(int(number/2), 1, kernel_size=1)
#         self.mix1=Attention(int(number/2),int(number/2))
#         self.mix2=Attention(int(number/2),int(number/2))
#         self.mix3=Attention(int(number/2),int(number/2))
#         self.mix4=Attention(int(number/2),int(number/2))
#         self.conv2x1=nn.Conv3d(int(number/2),1,kernel_size=1)
#         self.conv2x2=nn.Conv3d(int(number/2),1,kernel_size=1)
#         self.conv2x3=nn.Conv3d(int(number/2),1,kernel_size=1)
#         self.conv2x4=nn.Conv3d(int(number/2),1,kernel_size=1)
#         self.final=nn.Sequential(
#             nn.Conv3d(number*2, number*2, kernel_size=3, padding=1),
#             nn.GroupNorm(num_groups=8, num_channels=number*2, eps=1e-01, affine=True),
#             nn.PReLU(),
#             nn.Conv3d(number*2, 1, kernel_size=1))
        

#         for m in self.modules():
#             if isinstance(m,(nn.Conv3d,nn.ConvTranspose3d,nn.Linear)):
#                 nn.init.xavier_normal_(m.weight, gain=1.0)
#             elif isinstance(m,(nn.BatchNorm3d,nn.GroupNorm)):
#                 nn.init.constant_(m.weight,1)
#                 nn.init.constant_(m.bias,0)

#     def forward(self,x):
#         conv1=self.conv1(x)
#         pool1=torch.cat([x,conv1],dim=1)
#         pool1=self.maxpool(pool1)

#         conv2=self.conv2(pool1)
#         pool2=torch.cat([pool1,conv2],dim=1)
#         pool2=self.maxpool(pool2)

#         conv3=self.conv3(pool2)
#         pool3=torch.cat([pool2,conv3],dim=1)
#         pool3=self.maxpool(pool3)

#         conv4=self.conv4(pool3)
#         pool4=torch.cat([pool3,conv4],dim=1)
#         pool4=self.maxpool(pool4)

#         conv5=self.conv5(pool4)
#         conv5=torch.cat([pool4,conv5],dim=1)

#         up6=torch.cat([self.up6(conv5),conv4],dim=1)
#         conv6=self.conv6(up6)
#         conv6=torch.cat([up6,conv6],dim=1)

#         up7=torch.cat([self.up7(conv6),conv3],dim=1)
#         conv7=self.conv7(up7)       
#         conv7=torch.cat([up7,conv7],dim=1)


#         up8=torch.cat([self.up8(conv7),conv2],dim=1)
#         conv8=self.conv8(up8)
#         conv8=torch.cat([up8,conv8],dim=1)


#         up9=torch.cat([self.up9(conv8),conv1],dim=1)
#         conv9=self.conv9(up9)
#         conv9=torch.cat([up9,conv9],dim=1)

#         fs1=F.interpolate(self.featureScale1(conv9),x.size()[2:],mode='trilinear')
#         fs2=F.interpolate(self.featureScale2(conv8),x.size()[2:],mode='trilinear')
#         fs3=F.interpolate(self.featureScale3(conv7),x.size()[2:],mode='trilinear')
#         fs4=F.interpolate(self.featureScale4(conv6),x.size()[2:],mode='trilinear')
        
#         fs1o=self.conv1x1(fs1)
#         fs2o=self.conv1x2(fs2)
#         fs3o=self.conv1x3(fs3)
#         fs4o=self.conv1x4(fs4)
        
#         mix1,att1=self.mix1(fs1)
#         mix2,att2=self.mix2(fs2)
#         mix3,att3=self.mix3(fs3)
#         mix4,att4=self.mix4(fs4)
        
#         mix_out1=self.conv2x1(mix1)
#         mix_out2=self.conv2x2(mix2)
#         mix_out3=self.conv2x3(mix3)
#         mix_out4=self.conv2x4(mix4)
#         out=torch.cat((mix1,mix2,mix3,mix4),dim=1)
#         out=self.final(out)
#         if self.phase=='train':
#             return {'out':out,'as1':mix_out1,'as2':mix_out2,'as3':mix_out3,'as4':mix_out4,'s1':fs1o,'s2':fs2o,'s3':fs3o,'s4':fs4o}
#         else:
#             return out

# import numpy as np
# def dice_coef(y_true, y_pred):
#     smooth = 1e-5
#     y_true_f = y_true.reshape(len(y_true),-1)
#     y_pred_f = y_pred.reshape(len(y_true),-1)
#     intersection = np.sum(y_true_f * y_pred_f,axis=1)

#     dice=(2.*intersection + smooth) / (np.sum(y_true_f,axis=1) + np.sum(y_pred_f,axis=1) + smooth)
#     return np.sum(dice)/(len(y_true))

# device='cpu'
# if device=='cuda':
#     half='half'
# else:
#     half='float'
# if half=='half':
#     model=model().half().to(device)
# else:
#     model=model(number=32).to(device)
# # model.load_state_dict(torch.load('../model/o150le-2.pkl'))
# TTA=False
# model.load_state_dict(torch.load('../multiscale.pth.gz'))
# import glob
# import SimpleITK as sitk
# file=glob.glob('../../qinghua/test/data/*')
# dicep=[]
# SENS=[]
# SPEC=[]
# PREC=[]
# BACC=[]
# AUC=[]
# KAP=[]
# VS=[]
# NMCC=[]
# AHD=[]

# meana=[]
# stda=[]
# model.eval()

# from miseval import evaluate
# if TTA:
#     import ttach as tta
#     transforms=tta.Compose([
#             # tta.HorizontalFlip(),
#             tta.VerticalFlip(),
#             # tta.Rotate90(angles=[90,180,270])
#             # tta.Resize(sizes=[480]),       
#         ])
#     tta_model=tta.SegmentationTTAWrapper(model, transforms,merge_mode='tsharpen')
#     tta_model.eval()

# import pickle
# f=open('../test/testi.pkl','rb')
# testi=pickle.load(f)
# f.close()
# f=open('../test/testm.pkl','rb')
# testm=pickle.load(f)
# f.close()

# for zz in tqdm(range(len(file))):
# # for zz in tqdm(range(1,2)):
#     image=[]
#     mask=[]
#     imagefile=[]
#     maskfile=[]
#     mfile=glob.glob(file[zz]+'/*.nii.gz')
#     if len(mfile)==0:
#         mfile=glob.glob(file[zz]+'/*.nii')
#     maskfile.append(mfile[0])
        
#     names=glob.glob(file[zz]+'/*.dcm')
#     f=file[zz].replace('data','brain_data')
#     iii=sitk.ReadImage(f+'/image.nii.gz') 
#     m=sitk.ReadImage(maskfile[0])
#     ii=testi[zz]
#     mm=testm[zz]
#     mm=mm[0,0]
#     if half=='half':
#         ii=torch.as_tensor(ii).type(torch.float16).to(device)
#     else:
#         ii=torch.as_tensor(ii).float().to(device)
    
#     if TTA:
#         with torch.no_grad():
#             if ii.size(2)!=128:
#                 pred=model(ii)
    
#             else:
#                 pred=tta_model(ii)
#         if half=='half':
#             imgs_mask_test = pred.cpu().numpy()
    
#         else:
#             if type(pred)==dict:
#                 imgs_mask_test = pred['out'].numpy()
#             else:
#                 imgs_mask_test = pred.numpy()
        
#     else:
#         with torch.no_grad():
#             pred=model(ii)       
#             pred=pred['out'] 
#         if half=='half':
#             imgs_mask_test = F.sigmoid(pred).cpu().numpy()
    
#         else:
#             imgs_mask_test = F.sigmoid(pred).numpy()
    
    
#     imgs_mask_test[imgs_mask_test>=0.5]=1
#     imgs_mask_test[imgs_mask_test<0.5]=0
#     imgs_mask_test=imgs_mask_test
#     labels1=measure.label(imgs_mask_test[0,0,:,:,:])
#     region1=measure.regionprops(labels1)
#     unitVol=np.prod(m.GetSpacing())
#     for qq in range(len(region1)):
#         bbox=region1[qq].bbox
#         counts=np.sum(imgs_mask_test[0,0,bbox[0]:bbox[-3],bbox[1]:bbox[-2],bbox[2]:bbox[-1]])
#         if counts*unitVol<=5:
#             imgs_mask_test[0,0,bbox[0]:bbox[-3],bbox[1]:bbox[-2],bbox[2]:bbox[-1]]=0
#         if 5<=counts*unitVol<=70:
#             xa=int((bbox[0]+bbox[-3])/2)
#             xb=int((bbox[1]+bbox[-2])/2)
#             xc=int((bbox[2]+bbox[-1])/2)
#             if xa-20<0:
#                 aimg=ii[:,:,0:40, xb-32:xb+32,xc-32:xc+32]
#                 augmented_image=torch.nn.functional.interpolate(aimg,size=(80,128,128),mode='trilinear')
#                 with torch.no_grad():
#                     pred=model(augmented_image)       
#                     pred=pred['out'] 
#                 if half=='half':
#                     imgs_mask_test1 = F.sigmoid(pred).cpu()
            
#                 else:
#                     imgs_mask_test1 = F.sigmoid(pred)
                
#                 imgs_mask_test1=torch.nn.functional.interpolate(imgs_mask_test1,size=(40,64,64),mode='trilinear').numpy()
#                 imgs_mask_test1[imgs_mask_test1>=0.5]=1
#                 imgs_mask_test1[imgs_mask_test1<0.5]=0
#                 labels11=measure.label(imgs_mask_test1[0,0,:,:,:])
#                 region11=measure.regionprops(labels11)
#                 for jjj in range(len(region11)):
#                     bboxxx=region11[jjj].bbox
#                     counts11=np.sum(imgs_mask_test1[0,0,bboxxx[0]:bboxxx[-3],bboxxx[1]:bboxxx[-2],bboxxx[2]:bboxxx[-1]])
#                     if counts11*unitVol<=5:
#                         imgs_mask_test1[0,0,bbox[0]:bbox[-3],bbox[1]:bbox[-2],bbox[2]:bbox[-1]]=0
#                 imgs_mask_test[:,:,0:40, xb-32:xb+32,xc-32:xc+32]=imgs_mask_test1
#                 print('TTA with resize')
#             elif xb+32>384:
#                 aimg=ii[:,:,0:40, 320:384,xc-32:xc+32]
#                 augmented_image=torch.nn.functional.interpolate(aimg,size=(80,128,128),mode='trilinear')
#                 with torch.no_grad():
#                     pred=model(augmented_image)       
#                     pred=pred['out'] 
#                 if half=='half':
#                     imgs_mask_test1 = F.sigmoid(pred).cpu()
            
#                 else:
#                     imgs_mask_test1 = F.sigmoid(pred)
                
#                 imgs_mask_test1=torch.nn.functional.interpolate(imgs_mask_test1,size=(40,64,64),mode='trilinear').numpy()
#                 imgs_mask_test1[imgs_mask_test1>=0.5]=1
#                 imgs_mask_test1[imgs_mask_test1<0.5]=0
#                 labels11=measure.label(imgs_mask_test1[0,0,:,:,:])
#                 region11=measure.regionprops(labels11)
#                 for jjj in range(len(region11)):
#                     bboxxx=region11[jjj].bbox
#                     counts11=np.sum(imgs_mask_test1[0,0,bboxxx[0]:bboxxx[-3],bboxxx[1]:bboxxx[-2],bboxxx[2]:bboxxx[-1]])
#                     if counts11*unitVol<=5:
#                         imgs_mask_test1[0,0,bbox[0]:bbox[-3],bbox[1]:bbox[-2],bbox[2]:bbox[-1]]=0
#                 imgs_mask_test[:,:,0:40, 320:384,xc-32:xc+32]=imgs_mask_test1
#                 print('TTA with resize')
#             else:
#                 aimg=ii[:,:,xa-20:xa+20, xb-32:xb+32,xc-32:xc+32]
#                 augmented_image=torch.nn.functional.interpolate(aimg,size=(80,128,128),mode='trilinear')
#                 with torch.no_grad():
#                     pred=model(augmented_image)       
#                     pred=pred['out'] 
#                 if half=='half':
#                     imgs_mask_test1 = F.sigmoid(pred).cpu()
            
#                 else:
#                     imgs_mask_test1 = F.sigmoid(pred)
                
#                 imgs_mask_test1=torch.nn.functional.interpolate(imgs_mask_test1,size=(40,64,64),mode='trilinear').numpy()
#                 imgs_mask_test1[imgs_mask_test1>=0.5]=1
#                 imgs_mask_test1[imgs_mask_test1<0.5]=0
#                 labels11=measure.label(imgs_mask_test1[0,0,:,:,:])
#                 region11=measure.regionprops(labels11)
#                 for jjj in range(len(region11)):
#                     bboxxx=region11[jjj].bbox
#                     counts11=np.sum(imgs_mask_test1[0,0,bboxxx[0]:bboxxx[-3],bboxxx[1]:bboxxx[-2],bboxxx[2]:bboxxx[-1]])
#                     if counts11*unitVol<=5:
#                         imgs_mask_test1[0,0,bbox[0]:bbox[-3],bbox[1]:bbox[-2],bbox[2]:bbox[-1]]=0
#                 imgs_mask_test[:,:,xa-20:xa+20, xb-32:xb+32,xc-32:xc+32]=imgs_mask_test1
#                 print('TTA with resize')
    
#     dice=dice_coef(imgs_mask_test,mm)
#     sens=evaluate(mm,imgs_mask_test[0,0],metric='SENS')
#     spec=evaluate(mm,imgs_mask_test[0,0],metric='SPEC')
#     prec=evaluate(mm,imgs_mask_test[0,0],metric='PREC')
#     bacc=evaluate(mm,imgs_mask_test[0,0],metric='BACC')
#     auc=evaluate(mm,imgs_mask_test[0,0],metric='AUC')
#     kap=evaluate(mm,imgs_mask_test[0,0],metric='KAP')
#     vs=evaluate(mm,imgs_mask_test[0,0],metric='VS')
#     nmcc=evaluate(mm,imgs_mask_test[0,0],metric='nMCC')
#     ahd=0
#     for iii in range(imgs_mask_test.shape[2]):
#         ahd +=evaluate(mm[iii],imgs_mask_test[0,0,iii],metric='AHD')
#     ahd=ahd/imgs_mask_test.shape[2]
#     print('Filename:{}\nDice:{}'.format(file[zz],dice))
#     dicep.append(dice)
#     SENS.append(sens)
#     SPEC.append(spec)
#     PREC.append(prec)
#     BACC.append(bacc)
#     AUC.append(auc)
#     KAP.append(kap)
#     VS.append(vs)
#     NMCC.append(nmcc)
#     AHD.append(ahd)
# print('Dice:{}\nSensitivity:{}\nSpecificity:{}\nPrecision:{}\n\
# Banlanced ACC:{}\nAUC:{}\nKAP:{}\nVolumetric Similarity:{}\nNormalized Matthews Correlation Coefficient:{}\n\
# Averaged Hf:{}\n'.format(np.mean(dicep),np.mean(SENS),np.mean(SPEC),np.mean(PREC),np.mean(BACC),np.mean(AUC),np.mean(KAP),\
#               np.mean(VS),np.mean(NMCC),np.mean(AHD))) 
    
    #%%
# from torch.utils.data import DataLoader
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torchsummary import summary
# import torch.optim as optim
# from torch.optim import lr_scheduler

# import time
# from tqdm import tqdm
# from visdom import Visdom
# from tensorboardX import SummaryWriter
# import torchvision.utils as vutils
# import os
# from skimage import measure

# os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
# SumWriter=SummaryWriter(log_dir="./log")
# viz=Visdom()

# def conv_block(inc,out):
#     return nn.Sequential(
#         nn.Conv3d(in_channels=inc,out_channels=out,kernel_size=3,padding=1),
#         nn.GroupNorm(num_groups=8, num_channels=out, eps=1e-01, affine=True),
#         nn.PReLU(),
#         nn.Conv3d(in_channels=out,out_channels=out,kernel_size=3,padding=1),
#         nn.GroupNorm(num_groups=8, num_channels=out, eps=1e-01, affine=True),
#         nn.PReLU()
#         )

# def up_conv(inc,out):
#     return nn.Sequential(
#         nn.ConvTranspose3d(in_channels=inc,out_channels=out,kernel_size=2,stride=2),
#         nn.GroupNorm(num_groups=8, num_channels=out, eps=1e-01, affine=True),
#         nn.PReLU())
 
# def featureScale(ins, out):
#     return nn.Sequential(
#         nn.Conv3d(ins, out, 1, bias=False),
#         nn.GroupNorm(num_groups=8, num_channels=out, eps=1e-01, affine=True),
#         nn.PReLU()
#     )



# class groupBlock(nn.Module):
#     def __init__(self,inc,outc):
#         super(groupBlock,self).__init__()
#         d=[1,2,3,4]
#         self.inChannels=inc//4
#         self.outChannels=inc//4
#         self.grouped=nn.ModuleList()
#         for i in range(4):
#             self.grouped.append(
#                 nn.Sequential(
#                     nn.Conv3d(
#                         self.inChannels,
#                         self.outChannels,3,padding=d[i],dilation=d[i],bias=True),
#                         nn.GroupNorm(num_groups=2, num_channels=self.outChannels, eps=1e-01, affine=True),
#                         nn.PReLU()
#                     )
#                 )
    
#     def forward(self,x):
#         x_split=torch.split(x,self.inChannels,dim=1)
#         x=[conv(t) for conv,t in zip(self.grouped,x_split)]
#         x=torch.cat(x,dim=1)
#         return x

# class Attention(nn.Module):
#     """
#     modified from https://github.com/wulalago/FetalCPSeg/Network/MixAttNet.py
#     """
#     def __init__(self,inc,outc):
#         super(Attention,self).__init__()
#         self.group=groupBlock(inc,outc)
#         self.conv1=nn.Conv3d(outc,outc,kernel_size=1)
#         self.group1=groupBlock(outc,outc)
#         self.conv2=nn.Conv3d(outc,outc,kernel_size=1)
#         self.norm=nn.GroupNorm(num_groups=8, num_channels=outc, eps=1e-01, affine=True)
#         self.norm1=nn.GroupNorm(num_groups=8, num_channels=outc, eps=1e-01, affine=True)
#         self.act=nn.PReLU()
    
#     def forward(self,x):
#         group1=self.conv1(self.group(x))
#         group2=self.conv2(group1)
#         att=F.sigmoid(self.conv2(group2))
#         out=self.norm(x*att)+self.norm1(x)
#         return self.act(out),att
    
# class model(nn.Module):
#     def __init__(self,number=32,phase='train'):
#         super().__init__()
#         self.phase=phase
#         self.conv1=conv_block(1,number)
#         self.conv2=conv_block(number+1,number*2)
#         self.conv3=conv_block(number+1+number*2,number*4)
#         self.conv4=conv_block(number+1+number*2+number*4,number*8)
#         self.conv5=conv_block(number+1+number*2+number*4+number*8,number*16)
#         self.up6=up_conv(number+1+number*2+number*4+number*8+number*16,number*8)
#         self.conv6=conv_block(number*16,number*8)
#         self.up7=up_conv(number*16+number*8,number*4)
#         self.conv7=conv_block(number*8,number*4)
#         self.up8=up_conv(number*8+number*4,number*2)
#         self.conv8=conv_block(number*4,number*2)
#         self.up9=up_conv(number*4+number*2,number)
#         self.conv9=conv_block(number*2,number)
#         self.maxpool=nn.MaxPool3d(2)
#         self.featureScale1=featureScale(number*3,int(number/2))
#         self.featureScale2=featureScale(number*6,int(number/2))
#         self.featureScale3=featureScale(number*12,int(number/2))
#         self.featureScale4=featureScale(number*24,int(number/2))
#         self.conv1x1=nn.Conv3d(int(number/2),1, kernel_size=1)
#         self.conv1x2=nn.Conv3d(int(number/2), 1, kernel_size=1)
#         self.conv1x3=nn.Conv3d(int(number/2), 1, kernel_size=1)
#         self.conv1x4=nn.Conv3d(int(number/2), 1, kernel_size=1)
#         self.mix1=Attention(int(number/2),int(number/2))
#         self.mix2=Attention(int(number/2),int(number/2))
#         self.mix3=Attention(int(number/2),int(number/2))
#         self.mix4=Attention(int(number/2),int(number/2))
#         self.conv2x1=nn.Conv3d(int(number/2),1,kernel_size=1)
#         self.conv2x2=nn.Conv3d(int(number/2),1,kernel_size=1)
#         self.conv2x3=nn.Conv3d(int(number/2),1,kernel_size=1)
#         self.conv2x4=nn.Conv3d(int(number/2),1,kernel_size=1)
#         self.final=nn.Sequential(
#             nn.Conv3d(number*2, number*2, kernel_size=3, padding=1),
#             nn.GroupNorm(num_groups=8, num_channels=number*2, eps=1e-01, affine=True),
#             nn.PReLU(),
#             nn.Conv3d(number*2, 1, kernel_size=1))
        

#         for m in self.modules():
#             if isinstance(m,(nn.Conv3d,nn.ConvTranspose3d,nn.Linear)):
#                 nn.init.xavier_uniform_(m.weight, gain=1.0)
#             elif isinstance(m,(nn.BatchNorm3d,nn.GroupNorm)):
#                 nn.init.constant_(m.weight,1)
#                 nn.init.constant_(m.bias,0)

#     def forward(self,x):
#         conv1=self.conv1(x)
#         pool1=torch.cat([x,conv1],dim=1)
#         pool1=self.maxpool(pool1)

#         conv2=self.conv2(pool1)
#         pool2=torch.cat([pool1,conv2],dim=1)
#         pool2=self.maxpool(pool2)

#         conv3=self.conv3(pool2)
#         pool3=torch.cat([pool2,conv3],dim=1)
#         pool3=self.maxpool(pool3)

#         conv4=self.conv4(pool3)
#         pool4=torch.cat([pool3,conv4],dim=1)
#         pool4=self.maxpool(pool4)

#         conv5=self.conv5(pool4)
#         conv5=torch.cat([pool4,conv5],dim=1)

#         up6=torch.cat([self.up6(conv5),conv4],dim=1)
#         conv6=self.conv6(up6)
#         conv6=torch.cat([up6,conv6],dim=1)

#         up7=torch.cat([self.up7(conv6),conv3],dim=1)
#         conv7=self.conv7(up7)       
#         conv7=torch.cat([up7,conv7],dim=1)


#         up8=torch.cat([self.up8(conv7),conv2],dim=1)
#         conv8=self.conv8(up8)
#         conv8=torch.cat([up8,conv8],dim=1)


#         up9=torch.cat([self.up9(conv8),conv1],dim=1)
#         conv9=self.conv9(up9)
#         conv9=torch.cat([up9,conv9],dim=1)

#         fs1=F.interpolate(self.featureScale1(conv9),x.size()[2:],mode='trilinear')
#         fs2=F.interpolate(self.featureScale2(conv8),x.size()[2:],mode='trilinear')
#         fs3=F.interpolate(self.featureScale3(conv7),x.size()[2:],mode='trilinear')
#         fs4=F.interpolate(self.featureScale4(conv6),x.size()[2:],mode='trilinear')
        
#         fs1o=self.conv1x1(fs1)
#         fs2o=self.conv1x2(fs2)
#         fs3o=self.conv1x3(fs3)
#         fs4o=self.conv1x4(fs4)
        
#         mix1,att1=self.mix1(fs1)
#         mix2,att2=self.mix2(fs2)
#         mix3,att3=self.mix3(fs3)
#         mix4,att4=self.mix4(fs4)
        
#         mix_out1=self.conv2x1(mix1)
#         mix_out2=self.conv2x2(mix2)
#         mix_out3=self.conv2x3(mix3)
#         mix_out4=self.conv2x4(mix4)
#         out=torch.cat((mix1,mix2,mix3,mix4),dim=1)
#         out=self.final(out)
#         if self.phase=='train':
#             return {'out':out,'as1':mix_out1,'as2':mix_out2,'as3':mix_out3,'as4':mix_out4,'s1':fs1o,'s2':fs2o,'s3':fs3o,'s4':fs4o}
#         else:
#             return out

# import numpy as np
# def dice_coef(y_true, y_pred):
#     smooth = 1e-5
#     y_true_f = y_true.reshape(len(y_true),-1)
#     y_pred_f = y_pred.reshape(len(y_true),-1)
#     intersection = np.sum(y_true_f * y_pred_f,axis=1)

#     dice=(2.*intersection + smooth) / (np.sum(y_true_f,axis=1) + np.sum(y_pred_f,axis=1) + smooth)
#     return np.sum(dice)/(len(y_true))

# device='cpu'
# if device=='cuda':
#     half='half'
# else:
#     half='float'
# if half=='half':
#     model=model().half().to(device)
# else:
#     model=model().to(device)
# # model.load_state_dict(torch.load('../model/o150le-2.pkl'))
# model.load_state_dict(torch.load('../4.pth.gz'))
# import glob
# import SimpleITK as sitk
# file=glob.glob('../../qinghua/test/data/*')
# dicep=[]
# meana=[]
# stda=[]
# model.eval()

# import ttach as tta
# # transforms=tta.Compose([tta.HorizontalFlip(),tta.VerticalFlip(),tta.Rotate90(angles=[0,180])])
# transforms=tta.Compose([tta.Scale(scales=[2])])
# tta_model=tta.SegmentationTTAWrapper(model, transforms,merge_mode='mean')
# tta_model.eval()
# for zz in tqdm(range(28)):
# # for zz in tqdm(range(1,2)):
#     image=[]
#     mask=[]
#     imagefile=[]
#     maskfile=[]
#     mfile=glob.glob(file[zz]+'/*.nii.gz')
#     if len(mfile)==0:
#         mfile=glob.glob(file[zz]+'/*.nii')
#     maskfile.append(mfile[0])
        
#     names=glob.glob(file[zz]+'/*.dcm')
#     f=file[zz].replace('data','brain_data')
#     iii=sitk.ReadImage(f+'/image.nii.gz') 
#     ii=sitk.GetArrayFromImage(iii)
#     ii=np.clip(ii,0,2000)
#     m=sitk.ReadImage(maskfile[0])
#     mm=sitk.GetArrayFromImage(m)
    
   
#     if ii.shape[0]==mm.shape[0]:
#         pass
#     else:
#         ii=ii[0:96]
#         mm=mm[0:96]
    
#     sixteen=len(ii)/16
#     six=np.round(sixteen)
#     if six>sixteen:
#         slice=int(six*16)
#         var=slice-len(ii)
#         i2=np.zeros((slice,384,384))
#         m2=np.zeros((slice,384,384))
#         var2=var//2
#         i2[var2:len(ii)+var2]=ii[:,106:490,125:509]
#         m2[var2:len(ii)+var2]=mm[:,106:490,125:509]
#     elif six<sixteen:
#         slice=int(six*16)
#         var=len(ii)-slice
#         var2=var//2
#         i2=ii[var2:slice+var2,106:490,125:509]
#         m2=mm[var2:slice+var2,106:490,125:509]
#     else:
#         i2=ii[:,106:490,125:509]
#         m2=mm[:,106:490,125:509]
                 
#     img_rows = i2.shape[1]
#     img_cols = i2.shape[2]
#     img_depth = i2.shape[0]
#     ii=i2.reshape((1,1,img_depth,img_rows,img_cols))
#     mm=m2.reshape((1,1,img_depth,img_rows,img_cols)).astype(np.float64)

#     min=np.mean(ii)
#     max=np.std(ii)
#     ii=(ii-min)/(max)
    
#     if half=='half':
#         ii=torch.as_tensor(ii).type(torch.float16).to(device)
#     else:
#         ii=torch.as_tensor(ii).float().to(device)
    
#     with torch.no_grad():

#         if ii.size(2)!=128:
#             pred=model(ii)
#             pred=pred['out']

#         else:
#             pred=tta_model(ii)
 
#         # pred=model(ii)       
#         # pred=pred['out']
#     if half=='half':
#         imgs_mask_test = F.sigmoid(pred).cpu().numpy()
#     else:
#         imgs_mask_test = F.sigmoid(pred).numpy()
#     imgs_mask_test[imgs_mask_test>=0.5]=1
#     imgs_mask_test[imgs_mask_test<0.5]=0
#     imgs_mask_test=imgs_mask_test
#     labels1=measure.label(imgs_mask_test[0,0,:,:,:])
#     region1=measure.regionprops(labels1)
#     unitVol=np.prod(m.GetSpacing())
#     for qq in range(len(region1)):
#         bbox=region1[qq].bbox
#         counts=np.sum(imgs_mask_test[0,0,bbox[0]:bbox[-3],bbox[1]:bbox[-2],bbox[2]:bbox[-1]])

#         if counts*unitVol<=5:
#             imgs_mask_test[0,0,bbox[0]:bbox[-3],bbox[1]:bbox[-2],bbox[2]:bbox[-1]]=0
#     dice=dice_coef(imgs_mask_test,mm)
#     print(dice)
#     dicep.append(dice)

#%%
'''
特征可视化,得看特征到底有多少个，可以把所有特征都用visdom可视化出来
'''

# activation={}
# def get_activation(name):
#     def hook(model,input,output):
#         activation[name]=output.detach()
#     return hook

# model.conv9[3].register_forward_hook(get_activation("max"))
# with torch.no_grad():
#     pred,conc=model(ii)
# max=activation["max"]

# import matplotlib.pyplot as plt
# plt.figure(figsize=(16,8))
# for ii in range(128):
#     plt.subplot(8,16,ii+1)
#     plt.imshow(max.data.numpy()[0,1,ii,:,:],'gray')
#     plt.axis('off')
# plt.subplots_adjust(wspace=0.1,hspace=0.1)
# plt.show()
#%%
#尝试出现类似的热度图的结果


# with torch.no_grad():
#     pred,att=model(ii)

# d=(torch.sum(att,dim=1))/(att.size()[1])
# plt.figure(figsize=(16,8))
# for ii in range(128):
#     plt.subplot(8,16,ii+1)
#     plt.imshow(d[0,ii,:,:],'jet')
#     plt.axis('off')
# plt.subplots_adjust(wspace=0.1,hspace=0.1)
# plt.show()
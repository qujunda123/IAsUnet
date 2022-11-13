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

class models(nn.Module):
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

import SimpleITK as sitk
import mclahe as mc
import glob
import monai

device='cpu'
dicts=['32','clahe','64']
for kk in range(3):
    print('*'*30+'Inference for '+dicts[kk]+'*'*30)
     
    if dicts[kk]=='32':
        model=models(number=32,phase='val').to(device)
        wei='../32fu3103.pth.gz'
        model.load_state_dict(torch.load(wei)['model'])
        names='32'
    elif dicts[kk]=='clahe':
        model=models(number=32,phase='val').to(device)
        wei='../clahefu2.pth.gz'
        model.load_state_dict(torch.load(wei))
        names='clahe'
    else:
        model=models(number=64,phase='val').to(device)
        wei='../644503.pth.gz'
        model.load_state_dict(torch.load(wei))
        names='64'
        
    model.eval()
    file=glob.glob('./resample/*')
    
    for i in tqdm(range(len(file))):
        image=sitk.ReadImage(file[i])
        image_array=np.flip(sitk.GetArrayFromImage(image))
        image_arrayc=np.clip(image_array,0,2000)
    
        if 'clahe'==names:
            image_arrayc=mc.mclahe(image_arrayc,kernel_size=[16,84,84],n_bins=256,clip_limit=0.01)
        zs=int(image_array.shape[0]/2)
        hs=int(image_array.shape[1]/2)
        ws=int(image_array.shape[2]/2)
        image_arrayc=image_arrayc[zs-200:zs+100,hs-240:hs+294,ws-256:ws+256]
        image_arrayz=(image_arrayc-np.mean(image_arrayc))/np.std(image_arrayc)
        ii=torch.reshape(torch.as_tensor(image_arrayz),(1,1,image_arrayz.shape[0],image_arrayz.shape[1],image_arrayz.shape[2])).float().to(device)
        
        #阴性数据所扫描的范围过大，导致图像过大，因此改用滑窗法进行测试
        with torch.no_grad():
            pred=monai.inferers.sliding_window_inference(ii,(80,128,128),2,model,device='cpu')

        imgs_mask_test=F.sigmoid(pred).numpy()
        imgs_mask_test[imgs_mask_test>=0.5]=1
        imgs_mask_test[imgs_mask_test<0.5]=0
        labels1=measure.label(imgs_mask_test[0,0,:,:,:])
        region1=measure.regionprops(labels1)
        unitVol=np.prod(image.GetSpacing())
        for qq in range(len(region1)):
            bbox=region1[qq].bbox
            counts=np.sum(imgs_mask_test[0,0,bbox[0]:bbox[-3],bbox[1]:bbox[-2],bbox[2]:bbox[-1]])
            if counts*unitVol<=5:
                imgs_mask_test[0,0,bbox[0]:bbox[-3],bbox[1]:bbox[-2],bbox[2]:bbox[-1]]=0
        images=np.zeros(image_array.shape)
        images[zs-200:zs+100,hs-240:hs+294,ws-256:ws+256]=imgs_mask_test[0,0]
        np.save('./result/'+names+'-'+str(i)+'.npy',images)

#%%

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
# model=model(number=64,phase='val').to(device)
# wei='../644502.pth.gz'
# model.load_state_dict(torch.load(wei))
# # model.load_state_dict(torch.load(wei)['model'])
# names='64'

# model.eval()
# import SimpleITK as sitk
# import mclahe as mc
# import glob
# import monai


# file=glob.glob('./orginalData/*')

# for i in range(len(file)):
#     reader=sitk.ImageSeriesReader()
#     imgnames=reader.GetGDCMSeriesFileNames(file[i])
#     reader.SetFileNames(imgnames)
#     image=reader.Execute()
#     image_array=sitk.GetArrayFromImage(image)
#     image_arrayt=image_array.T#阴性是被旋转了，根据实际情况选择是否旋转。
#     image_arrayc=np.clip(image_arrayt,0,2000)

#     if 'clahe'==names:
#         image_arrayc=mc.mclahe(image_arrayc,kernel_size=[16,84,84],n_bins=256,clip_limit=0.01)
#     zs=int(image_array.shape[0]/2)
#     hs=int(image_array.shape[1]/2)
#     ws=int(image_array.shape[2]/2)
#     image_arrayc=image_arrayc[ws-192:ws+192,hs-128:hs+128,zs-128:zs+128]
#     image_arrayz=(image_arrayc-np.mean(image_arrayc))/np.std(image_arrayc)
#     ii=torch.reshape(torch.as_tensor(image_arrayz),(1,1,image_arrayz.shape[0],image_arrayz.shape[1],image_arrayz.shape[2])).float().to(device).permute(0,1,3,2,4)
   
#     #阴性数据所扫描的范围过大，导致图像过大，因此改用滑窗法进行测试
#     with torch.no_grad():
#         pred=monai.inferers.sliding_window_inference(ii,(80,128,128),2,model,device='cpu')
#     pred=pred.permute(0,1,3,2,4)
#     imgs_mask_test=F.sigmoid(pred).numpy()
#     imgs_mask_test[imgs_mask_test>=0.5]=1
#     imgs_mask_test[imgs_mask_test<0.5]=0
#     labels1=measure.label(imgs_mask_test[0,0,:,:,:])
#     region1=measure.regionprops(labels1)
#     unitVol=np.prod(image.GetSpacing())
#     for qq in range(len(region1)):
#         bbox=region1[qq].bbox
#         counts=np.sum(imgs_mask_test[0,0,bbox[0]:bbox[-3],bbox[1]:bbox[-2],bbox[2]:bbox[-1]])
#         if counts*unitVol<=5:
#             imgs_mask_test[0,0,bbox[0]:bbox[-3],bbox[1]:bbox[-2],bbox[2]:bbox[-1]]=0
#     images=np.zeros(image_arrayt.shape)
#     images[ws-192:ws+192,hs-128:hs+128,zs-128:zs+128]=imgs_mask_test[0,0]
#     images=images.T
#     np.save('./result/'+names+'-'+str(i)+'.npy',images)
    
    
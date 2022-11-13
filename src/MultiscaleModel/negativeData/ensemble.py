import numpy as np
import glob
from tqdm import tqdm
from skimage import measure

dicep=[]


def dice_coef(y_true, y_pred):
    smooth=1e-5
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)


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
from miseval import evaluate

predict=[]
file=glob.glob('orignialData/*')
for kk in range(len(file)):

    a=np.load('./result/'+'64-'+str(kk)+'.npy')
    b=np.load('./result/'+'32-'+str(kk)+'.npy')
    c=np.load('./result/'+'clahe-'+str(kk)+'.npy')

    add=np.zeros(a.shape)
    
    if np.all(a*b==0) and np.all(a*c==0) and np.all(c*b==0):
        predict.append(b)
        continue

    
    if np.all(a*c==0):
        pass
    else:
        bapp=[]
        minall=[]
        
        inter=a*c
        labels1=measure.label(inter)
        region1=measure.regionprops(labels1)
        all=a+b+c
        all[all>0.5]=1
        labels2=measure.label(all)
        region2=measure.regionprops(labels2)
        
        for qq in range(len(region1)):
            bbox=region1[qq].bbox
            bapp.append(bbox)
        
        indexbbox=[]
        for jjj in bapp:
            min1=[]
            for qq in range(len(region2)):
                bbox=region2[qq].bbox
                b2=np.abs(((bbox[-2]+bbox[1])/2)-((jjj[-2]+jjj[1]))/2)
                b3=np.abs(((bbox[-3]+bbox[2])/2)-((jjj[-3]+jjj[2]))/2)
                min1.append(np.abs((b2+b3)/2))
                indexbbox.append(bbox)
            
            min11=np.min(np.array(min1))
            minall.append(indexbbox[min1.index(min11)])

        for qq in range(len(minall)):
            bbox=minall[qq]
            add[bbox[0]:bbox[-3],bbox[1]:bbox[-2],bbox[2]:bbox[-1]]=all[bbox[0]:bbox[-3],bbox[1]:bbox[-2],bbox[2]:bbox[-1]]
        predict.append(add)
        continue
    
    if np.all(a*b==0):
        pass
    else:
        bapp=[]
        minall=[]
        
        inter=a*b
        labels1=measure.label(inter)
        region1=measure.regionprops(labels1)
        all=a+b+c
        all[all>0.5]=1
        labels2=measure.label(all)
        region2=measure.regionprops(labels2)
        
        for qq in range(len(region1)):
            bbox=region1[qq].bbox
            bapp.append(bbox)
        
        indexbbox=[]
        for jjj in bapp:
            min1=[]
            for qq in range(len(region2)):
                bbox=region2[qq].bbox
                b2=np.abs(((bbox[-2]+bbox[1])/2)-((jjj[-2]+jjj[1]))/2)
                b3=np.abs(((bbox[-3]+bbox[2])/2)-((jjj[-3]+jjj[2]))/2)
                min1.append(np.abs((b2+b3)/2))
                indexbbox.append(bbox)
            
            min11=np.min(np.array(min1))
            minall.append(indexbbox[min1.index(min11)])

        for qq in range(len(minall)):
            bbox=minall[qq]
            add[bbox[0]:bbox[-3],bbox[1]:bbox[-2],bbox[2]:bbox[-1]]=all[bbox[0]:bbox[-3],bbox[1]:bbox[-2],bbox[2]:bbox[-1]]
        predict.append(add)
        continue
    
    if np.all(b*c==0):
        pass
    else:
        bapp=[]
        minall=[]
        
        inter=c*b
        labels1=measure.label(inter)
        region1=measure.regionprops(labels1)
        all=a+b+c
        all[all>0.5]=1
        labels2=measure.label(all)
        region2=measure.regionprops(labels2)
        
        for qq in range(len(region1)):
            bbox=region1[qq].bbox
            bapp.append(bbox)
        
        indexbbox=[]
        for jjj in bapp:
            min1=[]
            for qq in range(len(region2)):
                bbox=region2[qq].bbox
                b2=np.abs(((bbox[-2]+bbox[1])/2)-((jjj[-2]+jjj[1]))/2)
                b3=np.abs(((bbox[-3]+bbox[2])/2)-((jjj[-3]+jjj[2]))/2)
                min1.append(np.abs((b2+b3)/2))
                indexbbox.append(bbox)
            
            min11=np.min(np.array(min1))
            minall.append(indexbbox[min1.index(min11)])

        for qq in range(len(minall)):
            bbox=minall[qq]
            add[bbox[0]:bbox[-3],bbox[1]:bbox[-2],bbox[2]:bbox[-1]]=all[bbox[0]:bbox[-3],bbox[1]:bbox[-2],bbox[2]:bbox[-1]]
        predict.append(add)
        continue

num=0
for i in range(len(predict)):
    add=predict[i]
    add[add>=0.5]=1
    add[add<0.5]=0
    labels1=measure.label(add)
    region1=measure.regionprops(labels1)
    num+=len(region1)
    print(len(region1))
print(num/len(predict))



#%%
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
file=glob.glob('./32fu/*')
for kk in tqdm(range(len(file))):

    a=np.load('./64fu/'+str(kk)+'.npy')
    b=np.load('./32fu/'+str(kk)+'.npy')
    c=np.load('./clahefu/'+str(kk)+'.npy')
    
    if a.shape[0]==a.shape[1]:
        a=a[0,0]
    if b.shape[0]==b.shape[1]:
        b=b[0,0]
    if c.shape[0]==c.shape[1]:
        c=c[0,0]
    
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
    

import SimpleITK as sitk
import glob

for i in tqdm(range(len(predict))):
    mask=np.load('./mask/'+str(i)+'.npy')
    add=predict[i]
    add[add>=0.5]=1
    add[add<0.5]=0
    dice=dice_coef(mask,add)

    dicep.append(dice)
    
    np.save('./ensemble/'+str(i)+'.npy',add)   
    niis=sitk.GetImageFromArray(add)
    sitk.WriteImage(niis,'./3Dplot/ensemble/'+str(i)+'.nii.gz')
    niim=sitk.GetImageFromArray(mask)
    sitk.WriteImage(niim,'./3Dplot/mask/'+str(i)+'m.nii.gz')
    
    sens=evaluate(mask[0,0],add,metric='SENS')
    spec=evaluate(mask[0,0],add,metric='SPEC')
    prec=evaluate(mask[0,0],add,metric='PREC')
    bacc=evaluate(mask[0,0],add,metric='BACC')
    auc=evaluate(mask[0,0],add,metric='AUC')
    kap=evaluate(mask[0,0],add,metric='KAP')
    vs=evaluate(mask[0,0],add,metric='VS')
    nmcc=evaluate(mask[0,0],add,metric='nMCC')
    ahd=0
    for iii in range(len(mask[0,0])):
        ahd +=evaluate(mask[0,0,iii],add[iii],metric='AHD')
    ahd=ahd/len(mask[0,0])
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
# import numpy as np
# import glob
# from tqdm import tqdm
# from skimage import measure
# dicep=[]

# def dice_coef(y_true, y_pred):
#     smooth = 1e-5
#     y_true_f = y_true.reshape(1,-1)
#     y_pred_f = y_pred.reshape(1,-1)
#     intersection = np.sum(y_true_f * y_pred_f,axis=1)
#     dice=(2.*intersection + smooth) / (np.sum(y_true_f,axis=1) + np.sum(y_pred_f,axis=1) + smooth)
#     return np.sum(dice)

# file=glob.glob('./clahe/*')
# for kk in tqdm(range(len(file))):
#     a=np.load('./64/'+str(kk)+'.npy')
#     b=np.load('./0.737/'+str(kk)+'.npy')
#     c=np.load('./clahe/'+str(kk)+'.npy')
#     if b.shape[0]==b.shape[1]==1:
#         b=b[0,0]
#     mask=np.load('./mask/'+str(kk)+'.npy')
 
    
#     labels1=measure.label(b)
#     region1=measure.regionprops(labels1)
#     unitVol=0.3*0.2976*0.2976

#     counts=np.sum(b)
#     if counts*unitVol<=100:
#         add=a+b
#         add[add>=0.5]=1
#     else:
#         add=b
    
#     dice=dice_coef(mask,add)
#     print(dice)
#     dicep.append(dice)
#     np.save('./ensemble/'+str(kk)+'.npy',add)
# #%%
# import glob
# import SimpleITK as sitk
# import numpy as np
# a=glob.glob('./mask/*')
# for i in range(len(a)):
#     image=np.load(a[0])
#     image=sitk.GetImageFromArray(image)
#     sitk.WriteImage(image,'./3Dplot/mask/'+str(i)+'m.nii.gz')
# #%%
# import SimpleITK as sitk
# predict=[]
# for kk in tqdm(range(20)):
    
#     a=np.load('./big IAs/64/'+str(kk)+'.npy')
#     b=np.load('./big IAs/0.737/'+str(kk)+'.npy')
#     c=np.load('./big IAs/clahe/'+str(kk)+'.npy')
    
#     if b.shape[0]==b.shape[1]:
#         b=b[0,0]
#     if c.shape[0]==c.shape[1]:
#         c=c[0,0]
    
#     add=np.zeros(a.shape)
    
#     if np.all(a*c==0):
#         pass
#     else:
#         bapp=[]
#         minall=[]
        
#         inter=a*c
#         labels1=measure.label(inter)
#         region1=measure.regionprops(labels1)
#         all=a+b+c
#         all[all>0.5]=1
#         labels2=measure.label(all)
#         region2=measure.regionprops(labels2)
        
#         for qq in range(len(region1)):
#             bbox=region1[qq].bbox
#             bapp.append(bbox)
        
#         indexbbox=[]
#         for jjj in bapp:
#             min1=[]
#             for qq in range(len(region2)):
#                 bbox=region2[qq].bbox
#                 b2=np.abs(((bbox[-2]+bbox[1])/2)-((jjj[-2]+jjj[1]))/2)
#                 b3=np.abs(((bbox[-3]+bbox[2])/2)-((jjj[-3]+jjj[2]))/2)
#                 min1.append(np.abs((b2+b3)/2))
#                 indexbbox.append(bbox)
            
#             min11=np.min(np.array(min1))
#             minall.append(indexbbox[min1.index(min11)])

#         for qq in range(len(minall)):
#             bbox=minall[qq]
#             add[bbox[0]:bbox[-3],bbox[1]:bbox[-2],bbox[2]:bbox[-1]]=all[bbox[0]:bbox[-3],bbox[1]:bbox[-2],bbox[2]:bbox[-1]]
#         predict.append(add)
#         continue
    
#     if np.all(b*c==0):
#         pass
#     else:
#         bapp=[]
#         minall=[]
        
#         inter=c*b
#         labels1=measure.label(inter)
#         region1=measure.regionprops(labels1)
#         all=a+b+c
#         all[all>0.5]=1
#         labels2=measure.label(all)
#         region2=measure.regionprops(labels2)
        
#         for qq in range(len(region1)):
#             bbox=region1[qq].bbox
#             bapp.append(bbox)
        
#         indexbbox=[]
#         for jjj in bapp:
#             min1=[]
#             for qq in range(len(region2)):
#                 bbox=region2[qq].bbox
#                 b2=np.abs(((bbox[-2]+bbox[1])/2)-((jjj[-2]+jjj[1]))/2)
#                 b3=np.abs(((bbox[-3]+bbox[2])/2)-((jjj[-3]+jjj[2]))/2)
#                 min1.append(np.abs((b2+b3)/2))
#                 indexbbox.append(bbox)
            
#             min11=np.min(np.array(min1))
#             minall.append(indexbbox[min1.index(min11)])

#         for qq in range(len(minall)):
#             bbox=minall[qq]
#             add[bbox[0]:bbox[-3],bbox[1]:bbox[-2],bbox[2]:bbox[-1]]=all[bbox[0]:bbox[-3],bbox[1]:bbox[-2],bbox[2]:bbox[-1]]
#         predict.append(add)
#         continue
    
#     if np.all(a*b==0):
#         pass
#     else:
#         bapp=[]
#         minall=[]
        
#         inter=a*b
#         labels1=measure.label(inter)
#         region1=measure.regionprops(labels1)
#         all=a+b+c
#         all[all>0.5]=1
#         labels2=measure.label(all)
#         region2=measure.regionprops(labels2)
        
#         for qq in range(len(region1)):
#             bbox=region1[qq].bbox
#             bapp.append(bbox)
        
#         indexbbox=[]
#         for jjj in bapp:
#             min1=[]
#             for qq in range(len(region2)):
#                 bbox=region2[qq].bbox
#                 b2=np.abs(((bbox[-2]+bbox[1])/2)-((jjj[-2]+jjj[1]))/2)
#                 b3=np.abs(((bbox[-3]+bbox[2])/2)-((jjj[-3]+jjj[2]))/2)
#                 min1.append(np.abs((b2+b3)/2))
#                 indexbbox.append(bbox)
            
#             min11=np.min(np.array(min1))
#             minall.append(indexbbox[min1.index(min11)])

#         for qq in range(len(minall)):
#             bbox=minall[qq]
#             add[bbox[0]:bbox[-3],bbox[1]:bbox[-2],bbox[2]:bbox[-1]]=all[bbox[0]:bbox[-3],bbox[1]:bbox[-2],bbox[2]:bbox[-1]]
#         predict.append(add)
#         continue
    
#     if np.all(a*b==0) and np.all(a*c==0) and np.all(c*b==0):
#         predict.append(b)
        
# for i in range(20):
#     add=predict[i]
#     add[add>=0.5]=1
#     add[add<0.5]=0
#     b=sitk.GetImageFromArray(add)
#     sitk.WriteImage(b,'./big IAs/ensemble/'+str(i)+'.nii.gz')


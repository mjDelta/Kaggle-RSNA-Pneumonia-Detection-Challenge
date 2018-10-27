# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 11:54:36 2018

@author: Mengji Zhang
"""
import pandas as pd
import os
import pydicom
from matplotlib import pyplot as plt
import numpy as np

###3 classes classification 
LABEL_NUMS=3
#path_labels="E:/Kaggle/RSNA Pneumonia Detection Challenge/stage_1_train_labels.csv"
#
#
#df_label=pd.read_csv(path_labels)
#df=pd.merge(df_class,df_label,on="patientId",how="inner")


#class_nums=set(list(df["class"]))
#print(class_nums)
#dir_path="E:/Kaggle/RSNA Pneumonia Detection Challenge/stage_1_test_images/"
#for p in os.listdir(dir_path):
#  full_path=os.path.join(dir_path,p)
#  ds=pydicom.read_file(full_path)
#  img=ds.pixel_array
#  plt.imshow(img,cmap=plt.get_cmap("gray_r"))
#  plt.show()
##  break

def dicom_to_array(path):
  dc=pydicom.read_file(path)
  return dc.pixel_array

def to_onehot(idxs):
  labels=np.zeros((len(idxs),LABEL_NUMS))
  for i in range(len(idxs)):
    labels[i,idxs[i]]=1
  return labels
  
###3 classes classification 
labels=["Lung Opacity","Normal","No Lung Opacity / Not Normal"]
path_class="E:/Kaggle/RSNA Pneumonia Detection Challenge/stage_1_detailed_class_info.csv"
train_path="E:/Kaggle/RSNA Pneumonia Detection Challenge/stage_1_train_images"
df_class=pd.read_csv(path_class)
ys=[]
xs=[]
for i,f in enumerate(os.listdir(train_path)):
  p_id=f[:-4]
  tmp=df_class[df_class["patientId"]==p_id]["class"].values[0]
  idx=labels.index(tmp)
  ys.append(idx)
  
  full_path=os.path.join(train_path,f)
  img=dicom_to_array(full_path)
  xs.append(img)
  plt.title(tmp)
  plt.imshow(img,cmap=plt.get_cmap("gray_r"))
  plt.show()
  if i==10:break
labels=to_onehot(ys)
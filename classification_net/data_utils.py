# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 14:25:21 2018

@author: Mengji Zhang
"""
import pydicom
import pandas as pd
import numpy as np
import os
from matplotlib import pyplot as plt
from scipy.misc import imresize
from tqdm import tqdm

CLF_NUMS=2

def dicom_to_array(path):
  dc=pydicom.read_file(path)
  return dc.pixel_array

def to_onehot(idxs):
  labels=np.zeros((len(idxs),CLF_NUMS))
  for i in range(len(idxs)):
    labels[i,idxs[i]]=1
  return labels
  
###2 classes classification generator:"Lung Opacity" vs ("Normal","No Lung Opacity / Not Normal")
def clf_generator(data_list,data_path,batch_size,CLF_SIZE):
  label="Lung Opacity"
  path_class="E:/Kaggle/RSNA Pneumonia Detection Challenge/stage_1_detailed_class_info.csv"
  df_class=pd.read_csv(path_class)


  batch_nums=len(data_list)//batch_size

  while True:
    for i in range(batch_nums):
      xs=[]
      ys=[]
      for f in data_list[i*batch_size:i*batch_size+batch_size]:
        p_id=f[:-4]
        tmp=df_class[df_class["patientId"]==p_id]["class"].values[0]
        idx=-1
        if tmp==label:
          idx=1
        else:
          idx=0
          
        full_path=os.path.join(data_path,f)
        img=dicom_to_array(full_path)
        img=imresize(img,(CLF_SIZE,CLF_SIZE))/255.
        
        xs.append(np.expand_dims(img,2))
        ys.append(idx)
      ys=to_onehot(ys)
      yield np.array(xs),ys

    if len(data_list)%batch_size!=0:
      xs=[]
      ys=[]
      for f in data_list[batch_nums*batch_size:]:
        p_id=f[:-4]
        tmp=df_class[df_class["patientId"]==p_id]["class"].values[0]
        idx=-1
        if tmp==label:
          idx=1
        else:
          idx=0
        
        full_path=os.path.join(data_path,f)
        img=dicom_to_array(full_path)
        img=imresize(img,(CLF_SIZE,CLF_SIZE))/255.

        xs.append(np.expand_dims(img,2))
        ys.append(idx)          
      ys=to_onehot(ys)
      yield np.array(xs),ys

def clf_steps(data_list,batch_size):
  num=len(data_list)//batch_size
  if len(data_list)%batch_size!=0:
    num+=1
  return num

def clf_test_data(path_list,parent_path,CLF_SIZE):
  xs=[]
  for path in path_list:
    full_path=os.path.join(parent_path,path)
    img=dicom_to_array(full_path)
    img=imresize(img,(CLF_SIZE,CLF_SIZE))/255.
    xs.append(np.expand_dims(img,2))  
  return np.array(xs)

if __name__=="__main__":
  train_path="E:/Kaggle/RSNA Pneumonia Detection Challenge/stage_1_train_images"

  train_list=os.listdir(train_path)
  clf_generator=clf_generator(train_list,train_path,6,256)
  steps=clf_steps(train_list,6)
  for s in tqdm(range(steps)):
    xs,ys=clf_generator.__next__()
    break

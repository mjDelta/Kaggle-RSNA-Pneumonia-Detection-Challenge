# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 14:36:20 2018

@author: Mengji Zhang
"""

from data_utils import clf_test_data
import os
import numpy as np  
import pandas as pd

def mkdir(path):
  if not os.path.exists(path):
    os.mkdir(path)
    
np.random.seed(1)

CLF_SIZE=256
model_index=3
TH=0.3
parent_path="E:/Kaggle/RSNA Pneumonia Detection Challenge/"
clf_dir=os.path.join(parent_path,"clf_outs")
mkdir(clf_dir)
data_path=os.path.join(parent_path,"stage_1_test_images")
if model_index==1:
  from models import VGG16
  model_dir=os.path.join(parent_path,"models_DOUBLECHANNELS_VGG16")
if model_index==2:
  from models import DENSE_VGG16 as VGG16
  model_dir=os.path.join(parent_path,"models_DENSE_VGG16")
if model_index==3:
  from models import VGG16_2 as VGG16
  model_dir=os.path.join(parent_path,"models_VGG16_2")
if model_index==4:
  from models import ResNet50 as VGG16
  BATCH_SIZE=32
  model_dir=os.path.join(parent_path,"models_RESNET50")
if model_index==5:
  from models import VGG16_3 as VGG16
  BATCH_SIZE=32
  model_dir=os.path.join(parent_path,"models_VGG16_3")  
models=["VGG16_0.h5","VGG16_1.h5","VGG16_2.h5"]

data_list=np.array(os.listdir(data_path))

avg_preds=np.zeros((len(data_list),))
test_x=clf_test_data(data_list,data_path,CLF_SIZE)

for model_p in models:

  model_path=os.path.join(model_dir,model_p) 
  model=VGG16(CLF_SIZE)
  
  model.load_weights(model_path)
  preds=model.predict(test_x,verbose=1)
  avg_preds+=preds[:,1]

avg_preds/=len(models)

opacity_path=os.path.join(clf_dir,"clf_opacity_paths.csv")
nonopacity_path=os.path.join(clf_dir,"clf_non_opacity_paths.csv")
opacity=[]
nonopacity=[]
for pred,data_path in zip(avg_preds,data_list):
  if pred>TH:
    opacity.append(data_path)
  else:
    nonopacity.append(data_path)
def to_csv(data,path):
  df=pd.DataFrame()
  df["path"]=data
  df.to_csv(path,index=False)

to_csv(opacity,opacity_path)
to_csv(nonopacity,nonopacity_path)
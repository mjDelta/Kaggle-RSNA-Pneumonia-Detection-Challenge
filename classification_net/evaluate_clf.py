# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 19:35:48 2018

@author: Mengji Zhang
"""
from data_utils import clf_generator
import os
import numpy as np  
import argparse

parser=argparse.ArgumentParser()
parser.add_argument("--model_index")
parser.add_argument("--threshold")
args=vars(parser.parse_args())
print(args)
  
np.random.seed(1)

CLF_SIZE=256
model_index=int(args["model_index"])
TH=float(args["threshold"])

parent_path="E:/Kaggle/RSNA Pneumonia Detection Challenge/"
data_path=os.path.join(parent_path,"stage_1_train_images")

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

chosed_idxs=np.random.choice(len(data_list),1000,replace=False)

val_list=data_list[chosed_idxs]

avg_preds=np.zeros((len(val_list),))
gen=clf_generator(val_list,data_path,len(val_list),CLF_SIZE)
val_x,val_y=gen.__next__()
for model_p in models:

  model_path=os.path.join(model_dir,model_p) 
  model=VGG16(CLF_SIZE)
  
  model.load_weights(model_path)
  preds=model.predict(val_x,verbose=1)
  avg_preds+=preds[:,1]

avg_preds/=len(models)

tp=0
tn=0
fp=0
fn=0
for pred,true in zip(avg_preds,val_y):
  if pred>TH and np.argmax(true)==1:
    tp+=1
  if pred<TH and np.argmax(true)==0:
    tn+=1
  if pred>TH and np.argmax(true)==0:
    fp+=1
  if pred<TH and np.argmax(true)==1:
    fn+=1  
print("=="*10)
print(model_dir)
print(models)
print("Threshold is :"+str(TH))
print("recall:"+str(tp/(tp+fn)))
print("precision:"+str(tp/(tp+fp+fn)))
print("acc:"+str((tp+tn)/(tp+fn+tn+fp)))

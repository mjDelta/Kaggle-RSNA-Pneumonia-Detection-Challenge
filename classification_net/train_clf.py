# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 15:14:30 2018

@author: Mengji Zhang
"""

from keras.callbacks import ModelCheckpoint,EarlyStopping
from data_utils import clf_generator,clf_steps
import os
import numpy as np
from sklearn.cross_validation import KFold
import argparse

def mkdir(path):
  if not os.path.exists(path):
    os.mkdir(path)

parser=argparse.ArgumentParser()
parser.add_argument("--model_index",type=int)
parser.add_argument("--CLF_SIZE",type=int)

args=vars(parser.parse_args())

model_index=args["model_index"] ###1: VGG16    2:DENSE_VGG16
CLF_SIZE=args["CLF_SIZE"]

np.random.seed(1)

EPOHCS=50
parent_path="E:/Kaggle/RSNA Pneumonia Detection Challenge/"
data_path=os.path.join(parent_path,"stage_1_train_images")

if model_index==1:
  from models import VGG16
  BATCH_SIZE=32
  model_dir=os.path.join(parent_path,"models_DOUBLECHANNELS_VGG16")
if model_index==2:
  from models import DENSE_VGG16 as VGG16
  BATCH_SIZE=32
  model_dir=os.path.join(parent_path,"models_DENSE_VGG16")
if model_index==3:
  from models import VGG16_2 as VGG16
  if CLF_SIZE==1024:
    
    BATCH_SIZE=1
  if CLF_SIZE==512:
    
    BATCH_SIZE=8 
  if CLF_SIZE==256:
    
    BATCH_SIZE=32
  model_dir=os.path.join(parent_path,"models_VGG16_2")

if model_index==4:
  from models import ResNet50 as VGG16
  BATCH_SIZE=16
  model_dir=os.path.join(parent_path,"models_RESNET50")
if model_index==5:
  from models import VGG16_3 as VGG16
  if CLF_SIZE==1024:
    
    BATCH_SIZE=1
  if CLF_SIZE==512:
    
    BATCH_SIZE=2 
  if CLF_SIZE==256:
    
    BATCH_SIZE=8
  model_dir=os.path.join(parent_path,"models_VGG16_3")
mkdir(model_dir)
print(model_dir)
data_list=np.array(os.listdir(data_path))

kf=KFold(len(data_list),n_folds=3,shuffle=True)

for i,(train_idxs,val_idxs) in enumerate(kf):
  train_list=data_list[train_idxs]
  val_list=data_list[val_idxs]

  model_path=os.path.join(model_dir,"CLF_SIZE"+str(CLF_SIZE)+"_VGG16_"+str(i)+".h5") 
  model=VGG16(CLF_SIZE)
  model.compile(optimizer="sgd",loss="binary_crossentropy",metrics=["acc"])
  model.fit_generator(clf_generator(train_list,data_path,BATCH_SIZE,CLF_SIZE),
                      steps_per_epoch=clf_steps(train_list,BATCH_SIZE),
                      epochs=EPOHCS,
                      validation_data=clf_generator(val_list,data_path,BATCH_SIZE,CLF_SIZE),
                      validation_steps=clf_steps(val_list,BATCH_SIZE),
                      callbacks=[ModelCheckpoint(model_path,monitor="val_loss",mode="min",save_best_only=True,verbose=1),
                                 EarlyStopping(monitor="val_loss",mode="min",patience=3)])
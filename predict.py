# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 12:32:14 2018

@author: Mengji Zhang
"""
from utils import get_test_data
from models import COORD_ASPP_UNET_DENSE as MODEL
import os
from skimage.morphology import remove_small_objects
from skimage.measure import label,regionprops
import numpy as np
import pandas as pd
from skimage.transform import resize
from matplotlib import pyplot as plt

SIZE=256
ORI_SIZE=1024
TH=0.5
parent_path="E:/Kaggle/RSNA Pneumonia Detection Challenge/"
model_dir=os.path.join(parent_path,"seg_models/COORD_ASPP_UNET_DENSE/")
models=["model_0.h5","model_1.h5","model_2.h5","model_3.h5","model_4.h5"]
#models=["model_0.h5","model_1.h5"]
test_dir="E:/Kaggle/RSNA Pneumonia Detection Challenge/stage_1_test_images/"
#test_dir="E:/Kaggle/RSNA Pneumonia Detection Challenge/test/"

test_x,patient_ids=get_test_data(test_dir,SIZE)
model=MODEL(SIZE)

avg_preds=np.zeros((len(test_x),SIZE,SIZE,1))

for model_p in models:

  model_path=os.path.join(model_dir,model_p) 
  model=MODEL(SIZE)
  
  model.load_weights(model_path)
  preds=model.predict(test_x,verbose=1)
  avg_preds+=preds

avg_preds/=len(models)



prediction_string=[]
for i,pred in enumerate(avg_preds):
  if i%100==0:
    print(i)
  pred=np.squeeze(pred)
  pred=resize(pred,(ORI_SIZE,ORI_SIZE),mode='reflect')
  pred_tmp=pred>TH
  pred_tmp=remove_small_objects(pred_tmp,800*4,connectivity=2)
  pred_tmp=label(pred_tmp)
#  plt.imshow(pred_tmp,cmap=plt.get_cmap("gray_r"))
#  plt.show()
  string=""
  for region in regionprops(pred_tmp):
    min_y,min_x,max_y,max_x=region.bbox
    
    confidence=np.mean(pred[min_y:max_y,min_x:max_x])
    string=string+str(confidence)+" "+str(min_x)+" "+str(min_y)+" "+" "+str((max_x-min_x))+" "+str((max_y-min_y))+" "
  string=string[:-1]
  prediction_string.append(string)

out_path=os.path.join(model_dir,"submisson-5-models.csv")
print(out_path)
df=pd.DataFrame()
df["patientId"]=patient_ids
df["PredictionString"]=prediction_string
df.to_csv(out_path,index=False)
    
from models import COORD_ASPP_UNET as MODEL
from utils import generator,get_opacity_list,IOU_numpy,draw_lines
import os
import numpy as np  
from matplotlib import pyplot as plt
from skimage.morphology import remove_small_objects
from skimage.measure import label,regionprops
np.random.seed(1)

CLF_SIZE=256
parent_path="E:/Kaggle/RSNA Pneumonia Detection Challenge/"
model_dir=os.path.join(parent_path,"seg_models/COORD_ASPP_UNET/")
models=["model_0.h5","model_1.h5","model_2.h5"]

data_list=get_opacity_list()

chosed_idxs=np.random.choice(len(data_list),5,replace=False)

val_list=data_list[chosed_idxs]

avg_preds=np.zeros((len(val_list),CLF_SIZE,CLF_SIZE,1))
gen=generator(len(val_list),val_list,CLF_SIZE)

val_x,val_y=gen.__next__()
for model_p in models:

  model_path=os.path.join(model_dir,model_p) 
  model=MODEL(CLF_SIZE)
  
  model.load_weights(model_path)
  preds=model.predict(val_x,verbose=1)
  avg_preds+=preds

avg_preds/=len(models)

avg_preds=avg_preds>0.2

remove_avg_preds=np.zeros((len(val_list),CLF_SIZE,CLF_SIZE,1))
for i,pred in enumerate(avg_preds):
  pred_tmp=remove_small_objects(pred,800,connectivity=2)
  pred_tmp=np.squeeze(pred_tmp)
  pred_tmp=label(pred_tmp)
  new_pred=np.zeros(pred_tmp.shape)
  for region in regionprops(pred_tmp):
    min_y,min_x,max_y,max_x=region.bbox
    new_pred[min_y:max_y,min_x:max_x]=1
    
  remove_avg_preds[i]=np.expand_dims(new_pred,2)


iou_coeffient=IOU_numpy(val_y,remove_avg_preds,CLF_SIZE)

for i,(pred,true,img) in enumerate(zip(remove_avg_preds,val_y,val_x)):
  path=os.path.join(parent_path,"seg_out/"+str(i)+".png")
  img=np.squeeze(img)
  true=np.squeeze(true)
  pred=np.squeeze(pred)
  '''plt.imshow(pred)
  plt.show()'''
  draw_lines(img,true,pred,path)
print("iou ： "+str(iou_coeffient))

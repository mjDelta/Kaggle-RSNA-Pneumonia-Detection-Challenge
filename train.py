from models import COORD_ASPP_UNET_RES as MODEL
from utils import generator_all_data,IOU,get_generator_steps,get_opacity_list_all_data,IOU_loss,get_one_train_x_y,draw_lines
from keras.callbacks import ModelCheckpoint,EarlyStopping
from sklearn.cross_validation import KFold
import os
from matplotlib import pyplot as plt
import numpy as np
def mkdir(path):
	if not os.path.exists(path):
		os.mkdir(path)

BATCH_SIZE=4
EPOCHS=10
SIZE=256
parent_path="E:/Kaggle/RSNA Pneumonia Detection Challenge/"
model_dir=os.path.join(parent_path,"seg_models/COORD_ASPP_UNET_RES_ALL_DATA/")
test_patient_id="00436515-870c-4b36-a041-de91049b9ab4"
test_x,test_y=get_one_train_x_y(test_patient_id,SIZE)
test_x=np.expand_dims(test_x,0);test_x=np.expand_dims(test_x,3)


mkdir(model_dir)

data_list=get_opacity_list_all_data()
print("Data Loaded...")

kf=KFold(len(data_list),n_folds=5,shuffle=True)
for i,(train_idxs,val_idxs) in enumerate(kf):
#	if i<4:continue
	model_path=os.path.join(model_dir,"model_"+str(i)+".h5")
	model=MODEL(SIZE)
	model.compile(loss="binary_crossentropy",optimizer="adam",metrics=["acc",IOU])
	if i==3:
		model.load_weights(model_path)

	print("KFold Model Loaded...")
	print(model_path)

	pred_y=model.predict(test_x)
	draw_lines(np.squeeze(test_x),test_y,np.squeeze(pred_y))
	train_list=data_list[train_idxs]
	val_list=data_list[val_idxs]

	
	model.fit_generator(generator_all_data(BATCH_SIZE,train_list,SIZE),
											steps_per_epoch=2*get_generator_steps(BATCH_SIZE,train_list),
											epochs=EPOCHS,
											validation_data=generator_all_data(BATCH_SIZE,val_list,SIZE),
											validation_steps=get_generator_steps(BATCH_SIZE,val_list),
											callbacks=[ModelCheckpoint(model_path,monitor="val_loss",mode="min",verbose=1,save_best_only=True),
																 EarlyStopping(monitor="val_loss",patience=3,mode="min")])
	pred_y=model.predict(test_x)
	draw_lines(np.squeeze(test_x),test_y,np.squeeze(pred_y))
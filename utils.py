import pandas as pd
import os
import pydicom
import cv2
from matplotlib import pyplot as plt
import numpy as np
from keras import backend as K
from scipy.misc import imresize
from skimage.measure import find_contours
from matplotlib import pyplot as plt

EPSILON=1e-7
def dicom_to_array(file_path):
	dc=pydicom.read_file(file_path)
	x=dc.pixel_array
	#x=x.astype(np.int8)
	return x

def generator(batch_size,data_list,size):
	parent_path="E:/Kaggle/RSNA Pneumonia Detection Challenge/"
	train_label_path=os.path.join(parent_path,"stage_1_train_labels.csv")

	df=pd.read_csv(train_label_path,header=0)
	while True:
		xs=[]
		ys=[]
		for full_path in data_list:
			f=full_path.split("\\")[-1]
			label=df[df["patientId"]==f[:-4]]

			if len(label)==1 and label.iloc[0,-1]==0:
				continue
			x=dicom_to_array(full_path)
			y=np.zeros(x.shape)
			for i in range(len(label)):
				l=label.iloc[i,:]
				xmin=int(l["x"]);ymin=int(l["y"]);width=int(l["width"]);height=int(l["height"])
				cv2.rectangle(y,(xmin,ymin),(xmin+width,ymin+height),1,-1)
    
			x=imresize(x,(size,size))/255.
			y=imresize(y,(size,size))/255.

			##horzatal flip as the data augmentation
			if np.random.random(1)[0]>0.5:
				x=np.fliplr(x)
				y=np.fliplr(y)
			x=np.expand_dims(x,2)
			y=np.expand_dims(y,2)
			xs.append(x)
			ys.append(y)
			if len(xs)==batch_size:
				yield np.array(xs),np.array(ys)
				xs=[]
				ys=[]
		if len(xs)!=0:
			yield np.array(xs),np.array(ys)
def generator_all_data(batch_size,data_list,size):
	parent_path="E:/Kaggle/RSNA Pneumonia Detection Challenge/"
	train_label_path=os.path.join(parent_path,"stage_1_train_labels.csv")

	df=pd.read_csv(train_label_path,header=0)
	while True:
		xs=[]
		ys=[]
		for full_path in data_list:
			f=full_path.split("\\")[-1]
			label=df[df["patientId"]==f[:-4]]


			x=dicom_to_array(full_path)
			y=np.zeros(x.shape)
			if not (label.iloc[0,-1]==0):
				
				for i in range(len(label)):
					l=label.iloc[i,:]
					xmin=int(l["x"]);ymin=int(l["y"]);width=int(l["width"]);height=int(l["height"])
					cv2.rectangle(y,(xmin,ymin),(xmin+width,ymin+height),1,-1)
    
			x=imresize(x,(size,size))/255.
			y=imresize(y,(size,size))/255.

			##horzatal flip as the data augmentation
			if np.random.random(1)[0]>0.5:
				x=np.fliplr(x)
				y=np.fliplr(y)
			x=np.expand_dims(x,2)
			y=np.expand_dims(y,2)
			xs.append(x)
			ys.append(y)
			if len(xs)==batch_size:
				yield np.array(xs),np.array(ys)
				xs=[]
				ys=[]
		if len(xs)!=0:
			yield np.array(xs),np.array(ys)
def get_one_train_x_y(patient_id,size):
	parent_path="E:/Kaggle/RSNA Pneumonia Detection Challenge/"
	train_label_path=os.path.join(parent_path,"stage_1_train_labels.csv")	
	df=pd.read_csv(train_label_path,header=0)
	label=df[df["patientId"]==patient_id]
	full_path=os.path.join(parent_path,"stage_1_train_images",patient_id+".dcm")
	x=dicom_to_array(full_path)
	y=np.zeros(x.shape)
	for i in range(len(label)):
		l=label.iloc[i,:]
		xmin=int(l["x"]);ymin=int(l["y"]);width=int(l["width"]);height=int(l["height"])
		cv2.rectangle(y,(xmin,ymin),(xmin+width,ymin+height),1,-1)
	x=imresize(x,(size,size))/255.
	y=imresize(y,(size,size))/255.
	return x,y
def get_test_data(test_dir,size):
  xs=[]
  patient_ids=[]
  for f in os.listdir(test_dir):
    full_path=os.path.join(test_dir,f)
    x=dicom_to_array(full_path)
    x=imresize(x,(size,size))/255.
    x=np.expand_dims(x,2)
    xs.append(x)
    patient_ids.append(f[:-4])
  return np.array(xs),np.array(patient_ids)

def get_generator_steps(batch_size,data_list):
	counter=len(data_list)
	steps=counter//batch_size
	if counter%batch_size!=0:
		steps+=1
	return steps

def get_opacity_list():
	parent_path="E:/Kaggle/RSNA Pneumonia Detection Challenge/"
	train_label_path=os.path.join(parent_path,"stage_1_train_labels.csv")
#	train_imgs_dir=os.path.join(parent_path,"test")
	train_imgs_dir=os.path.join(parent_path,"stage_1_train_images")
	df=pd.read_csv(train_label_path,header=0)
	opacity=[]
	for f in os.listdir(train_imgs_dir):
		label=df[df["patientId"]==f[:-4]]
		if len(label)==1 and label.iloc[0,-1]==0:
			continue
		full_path=os.path.join(train_imgs_dir,f)
		opacity.append(full_path)
	return np.array(opacity)

def get_opacity_list_all_data():
	parent_path="E:/Kaggle/RSNA Pneumonia Detection Challenge/"
	train_label_path=os.path.join(parent_path,"stage_1_train_labels.csv")
#	train_imgs_dir=os.path.join(parent_path,"test")
	train_imgs_dir=os.path.join(parent_path,"stage_1_train_images")
	df=pd.read_csv(train_label_path,header=0)
	opacity=[]
	for f in os.listdir(train_imgs_dir):

		full_path=os.path.join(train_imgs_dir,f)
		opacity.append(full_path)
	return np.array(opacity)

def IOU(true,pred):
	true_flatten=K.clip(K.batch_flatten(true),0,1)
	pred_flatten=K.clip(K.batch_flatten(pred),0,1)

	true_flatten=K.greater(true_flatten,0.5)
	pred_flatten=K.greater(pred_flatten,0.5)
	true_flatten=K.cast(true_flatten,"float32")
	pred_flatten=K.cast(pred_flatten,"float32")

	intersection=K.sum(true_flatten*pred_flatten,axis=1)
	union=K.sum(K.maximum(true_flatten,pred_flatten),axis=1)+EPSILON

	#print(union.shape)
	#union=K.switch(K.equal(union,0),1,union)
	return K.mean(intersection/K.cast(union,"float32"))

def IOU_loss(true,pred):
	return 1.-IOU(true,pred)

def draw_lines(img,true,pred):
	true=true.astype("float32")
	pred=pred.astype("float32")
	true_contours=find_contours(true,0.5) 
	pred_contours=find_contours(pred,0.5)
	plt.imshow(img,cmap=plt.get_cmap("gray_r"))
	for i,contour in enumerate(true_contours):
		if i==0:
			plt.plot(contour[:,1],contour[:,0],linewidth=0.5,color="blue",label="true")
		else:
			plt.plot(contour[:,1],contour[:,0],linewidth=0.5,color="blue")

	for i,contour in enumerate(pred_contours):
		if i==0:
			plt.plot(contour[:,1],contour[:,0],linewidth=0.5,color="red",label="pred")
		else:
			plt.plot(contour[:,1],contour[:,0],linewidth=0.5,color="red")

	plt.legend()
	plt.show()




def IOU_numpy(true,pred,SIZE):
	true_flatten=np.zeros((len(true),SIZE*SIZE))
	pred_flatten=np.zeros((len(pred),SIZE*SIZE))

	for i,(t,p) in enumerate(zip(true,pred)):
		true_flatten[i]=t.flatten()
		pred_flatten[i]=p.flatten()

	true_flatten=true_flatten>0.5
	pred_flatten=pred_flatten>0.5

	intersection=np.sum(true_flatten*pred_flatten,axis=1)
	union=np.sum(np.maximum(true_flatten,pred_flatten),axis=1)+EPSILON	

	return np.mean(intersection/union)

if __name__=="__main__":
	data_list=get_opacity_list()
	gen=generator(2,data_list,256)
	x,y=gen.__next__()
	print(x.shape)
	print(y.shape)
	y0=np.squeeze(y[0])
	fig=plt.figure()
	fig.add_subplot(121)
	plt.imshow(np.squeeze(x[0]),cmap=plt.get_cmap("gray_r"))
	fig.add_subplot(122)
	plt.imshow(np.squeeze(y[0]),cmap=plt.get_cmap("gray_r"))
	plt.show()
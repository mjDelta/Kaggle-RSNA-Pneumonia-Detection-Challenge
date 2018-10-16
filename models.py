from coordConv import CoordinateChannel2D
from keras.layers import Conv2D,MaxPooling2D,UpSampling2D,Input,Activation,BatchNormalization,DepthwiseConv2D,AveragePooling2D,LeakyReLU
from keras.layers import concatenate,add
from keras.models import Model

def basic_conv(x,filters,activation,filter_size=(1,1)):
	x=Conv2D(filters,filter_size,padding="same",use_bias=False)(x)
	x=BatchNormalization()(x)
	x=Activation(activation)(x)
	return x

def basic_depth(x,strides=(2,2)):
	x=DepthwiseConv2D((3,3),padding="same",use_bias=False,strides=strides)(x)
	x=BatchNormalization()(x)
	x=Activation("relu")(x)
	return x

def basic_depth_conv(x,filters,activation="relu"):
	x=basic_depth(x,strides=(1,1))
	x=basic_conv(x,filters,activation)
	return x



def aspp(x,filters,size):
	b0=basic_conv(x,filters,activation="relu",filter_size=(1,1))

	b1=DepthwiseConv2D((3,3),dilation_rate=(6,6),padding="same",use_bias=False)(x)
	b1=BatchNormalization()(b1)
	b1=Activation("relu")(b1)
	b1=basic_conv(b1,filters,activation="relu")

	b2=DepthwiseConv2D((3,3),dilation_rate=(12,12),padding="same",use_bias=False)(x)
	b2=BatchNormalization()(b2)
	b2=Activation("relu")(b2)
	b2=basic_conv(b2,filters,activation="relu")

	b3=DepthwiseConv2D((3,3),dilation_rate=(24,24),padding="same",use_bias=False)(x)
	b3=BatchNormalization()(b3)
	b3=Activation("relu")(b3)
	b3=basic_conv(b3,filters,activation="relu")

	b4=AveragePooling2D((size,size))(x)
	b4=basic_conv(b4,filters,activation="relu")
	b4=UpSampling2D((size,size))(b4)

	out=concatenate([b0,b1,b2,b3,b4])
	return out

def COORD_ASPP_UNET(SIZE):
	size_=SIZE//16

	input_=Input(shape=(SIZE,SIZE,1))

	coord=CoordinateChannel2D(use_radius=True)(input_)
	encoder_conv1=basic_depth_conv(coord,32)
	encoder_pool1=basic_depth(encoder_conv1)

	encoder_conv2=basic_depth_conv(encoder_pool1,64)
	encoder_pool2=basic_depth(encoder_conv2)

	encoder_conv3=basic_depth_conv(encoder_pool2,128)
	encoder_pool3=basic_depth(encoder_conv3)

	encoder_conv4=basic_depth_conv(encoder_pool3,256)
	encoder_pool4=basic_depth(encoder_conv4)

	tmp=basic_depth_conv(encoder_pool4,256)
	tmp=aspp(tmp,256,size_)
	tmp=basic_depth_conv(tmp,256)

	decoder_up1=UpSampling2D()(tmp)
	decoder_con1=concatenate([decoder_up1,encoder_conv4])
	deconder_conv1=basic_depth_conv(decoder_con1,256)

	decoder_up2=UpSampling2D()(deconder_conv1)
	decoder_con2=concatenate([decoder_up2,encoder_conv3])
	deconder_conv2=basic_depth_conv(decoder_con2,128)

	decoder_up3=UpSampling2D()(deconder_conv2)
	decoder_con3=concatenate([decoder_up3,encoder_conv2])
	deconder_conv3=basic_depth_conv(decoder_con3,64)

	decoder_up4=UpSampling2D()(deconder_conv3)
	decoder_con4=concatenate([decoder_up4,encoder_conv1])
	deconder_conv4=basic_depth_conv(decoder_con4,32)

	out=basic_depth_conv(deconder_conv4,1,activation="sigmoid")

	model=Model(input=input_,output=out)
	return model

def COORD_ASPP_UNET_RES(SIZE,conv_depth=2):
	size_=SIZE//16

	input_=Input(shape=(SIZE,SIZE,1))

	coord=CoordinateChannel2D(use_radius=True)(input_)
	encoder_conv1=coord
	tmp=[]
	for i in range(conv_depth):
		encoder_conv1=basic_depth_conv(encoder_conv1,32)
		tmp.append(encoder_conv1)
	encoder_conv1=add(tmp)
	encoder_pool1=basic_depth(encoder_conv1)

	encoder_conv2=encoder_pool1
	tmp=[]
	for i in range(conv_depth):
		encoder_conv2=basic_depth_conv(encoder_conv2,64)
		tmp.append(encoder_conv2)
	encoder_conv2=add(tmp)
	encoder_pool2=basic_depth(encoder_conv2)

	encoder_conv3=encoder_pool2
	tmp=[]
	for i in range(conv_depth):
		encoder_conv3=basic_depth_conv(encoder_conv3,128)
		tmp.append(encoder_conv3)
	encoder_conv3=add(tmp)
	encoder_pool3=basic_depth(encoder_conv3)

	encoder_conv4=encoder_pool3
	tmp=[]
	for i in range(conv_depth):
		encoder_conv4=basic_depth_conv(encoder_conv4,256)
		tmp.append(encoder_conv4)
	encoder_conv4=add(tmp)
	encoder_pool4=basic_depth(encoder_conv4)

	tmp=basic_depth_conv(encoder_pool4,256)
	tmp=aspp(tmp,256,size_)
	tmp=basic_depth_conv(tmp,256)

	decoder_up1=UpSampling2D()(tmp)
	decoder_con1=concatenate([decoder_up1,encoder_conv4])
	decoder_conv1=decoder_con1
	tmp=[]
	for i in range(conv_depth):
		decoder_conv1=basic_depth_conv(decoder_conv1,256)
		tmp.append(decoder_conv1)
	decoder_conv1=add(tmp)

	decoder_up2=UpSampling2D()(decoder_conv1)
	decoder_con2=concatenate([decoder_up2,encoder_conv3])
	decoder_conv2=decoder_con2
	tmp=[]
	for i in range(conv_depth):
		decoder_conv2=basic_depth_conv(decoder_conv2,128)
		tmp.append(decoder_conv2)
	decoder_conv2=add(tmp)

	decoder_up3=UpSampling2D()(decoder_conv2)
	decoder_con3=concatenate([decoder_up3,encoder_conv2])
	decoder_conv3=decoder_con3
	tmp=[]
	for i in range(conv_depth):
		decoder_conv3=basic_depth_conv(decoder_conv3,64)
		tmp.append(decoder_conv3)
	decoder_conv3=add(tmp)

	decoder_up4=UpSampling2D()(decoder_conv3)
	decoder_con4=concatenate([decoder_up4,encoder_conv1])
	decoder_conv4=decoder_con4
	tmp=[]
	for i in range(conv_depth):
		decoder_conv4=basic_depth_conv(decoder_conv4,32)
		tmp.append(decoder_conv4)
	decoder_conv4=add(tmp)

	out=basic_depth_conv(decoder_conv4,1,activation="sigmoid")

	model=Model(input=input_,output=out)
	return model 

def COORD_ASPP_UNET_DENSE(SIZE,conv_depth=3):
	size_=SIZE//16

	input_=Input(shape=(SIZE,SIZE,1))

	coord=CoordinateChannel2D(use_radius=True)(input_)
	encoder_conv1=basic_depth_conv(coord,32)
	tmp=[encoder_conv1]
	for i in range(conv_depth):
		if i!=0:
			encoder_conv1=add(tmp)
		encoder_conv1=basic_depth_conv(encoder_conv1,32)
		tmp.append(encoder_conv1)
	encoder_pool1=basic_depth(encoder_conv1)

	encoder_conv2=basic_depth_conv(encoder_pool1,64)
	tmp=[encoder_conv2]
	for i in range(conv_depth):
		if i!=0:
			encoder_conv2=add(tmp)
		encoder_conv2=basic_depth_conv(encoder_conv2,64)
		tmp.append(encoder_conv2)
	encoder_pool2=basic_depth(encoder_conv2)

	encoder_conv3=basic_depth_conv(encoder_pool2,128)
	tmp=[encoder_conv3]
	for i in range(conv_depth):
		if i!=0:
			encoder_conv3=add(tmp)
		encoder_conv3=basic_depth_conv(encoder_conv3,128)
		tmp.append(encoder_conv3)
	encoder_pool3=basic_depth(encoder_conv3)

	encoder_conv4=basic_depth_conv(encoder_pool3,256)
	tmp=[encoder_conv4]
	for i in range(conv_depth):
		if i!=0:
			encoder_conv4=add(tmp)
		encoder_conv4=basic_depth_conv(encoder_conv4,256)
		tmp.append(encoder_conv4)
	encoder_pool4=basic_depth(encoder_conv4)

	tmp=basic_depth_conv(encoder_pool4,256)
	tmp=aspp(tmp,256,size_)
	tmp=basic_depth_conv(tmp,256)

	decoder_up1=UpSampling2D()(tmp)
	decoder_con1=concatenate([decoder_up1,encoder_conv4])
	decoder_conv1=basic_depth_conv(decoder_con1,256)
	tmp=[decoder_conv1]
	for i in range(conv_depth):
		if i!=0:
			decoder_conv1=add(tmp)
		decoder_conv1=basic_depth_conv(decoder_conv1,256)
		tmp.append(decoder_conv1)
	
	decoder_up2=UpSampling2D()(decoder_conv1)
	decoder_con2=concatenate([decoder_up2,encoder_conv3])
	decoder_conv2=basic_depth_conv(decoder_con2,128)
	tmp=[decoder_conv2]
	for i in range(conv_depth):
		if i!=0:
			decoder_conv2=add(tmp)
		decoder_conv2=basic_depth_conv(decoder_conv2,128)
		tmp.append(decoder_conv2)

	decoder_up3=UpSampling2D()(decoder_conv2)
	decoder_con3=concatenate([decoder_up3,encoder_conv2])
	decoder_conv3=basic_depth_conv(decoder_con3,64)
	tmp=[decoder_conv3]
	for i in range(conv_depth):
		if i!=0:
			decoder_conv3=add(tmp)
		decoder_conv3=basic_depth_conv(decoder_conv3,64)
		tmp.append(decoder_conv3)

	decoder_up4=UpSampling2D()(decoder_conv3)
	decoder_con4=concatenate([decoder_up4,encoder_conv1])
	decoder_conv4=basic_depth_conv(decoder_con4,32)
	tmp=[decoder_conv4]
	for i in range(conv_depth):
		if i!=0:
			decoder_conv4=add(tmp)
		decoder_conv4=basic_depth_conv(decoder_conv4,32)
		tmp.append(decoder_conv4)

	out=basic_depth_conv(decoder_conv4,1,activation="sigmoid")

	model=Model(input=input_,output=out)
	return model 

if __name__=="__main__":
	model=COORD_ASPP_UNET_RES(256)
	model.summary()
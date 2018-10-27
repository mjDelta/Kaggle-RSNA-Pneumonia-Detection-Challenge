# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 14:57:04 2018

@author: Mengji Zhang
"""

from keras.layers import Input,Conv2D,MaxPooling2D,Flatten,Dense,BatchNormalization,Activation,AveragePooling2D,Add,ZeroPadding2D
from keras.models import Model
from keras.layers import concatenate
from keras.initializers import glorot_uniform

def basic_conv(x,filters):
  x=Conv2D(filters,(3,3),padding="same")(x)
  x=BatchNormalization()(x)
  x=Activation("relu")(x)
  return x
  
def VGG16(CLF_SIZE):
  input_=Input(shape=(CLF_SIZE,CLF_SIZE,1))
  
  conv1=basic_conv(input_,32)
  conv1=basic_conv(conv1,32)
  pool1=MaxPooling2D()(conv1)
  
  conv2=basic_conv(pool1,32)
  conv2=basic_conv(conv2,32)
  pool2=MaxPooling2D()(conv2)
  
  conv3=basic_conv(pool2,64)
  conv3=basic_conv(conv3,64)
  conv3=basic_conv(conv3,64)
  pool3=MaxPooling2D()(conv3)
  
  conv4=basic_conv(pool3,64)
  conv4=basic_conv(conv4,64)
  conv4=basic_conv(conv4,64)
  pool4=MaxPooling2D()(conv4)
  
  conv5=basic_conv(pool4,64)
  conv5=basic_conv(conv5,64)
  conv5=basic_conv(conv5,64)
  pool5=MaxPooling2D()(conv5)
  
  flatten=Flatten()(pool5)
  
  dense1=Dense(512,activation="relu")(flatten)
  dense2=Dense(256,activation="relu")(dense1)
  dense3=Dense(2,activation="sigmoid")(dense2)
  
  model=Model(input=input_,output=dense3)
  return model
def VGG16_2(CLF_SIZE):
  input_=Input(shape=(CLF_SIZE,CLF_SIZE,1))
  
  conv1_1=basic_conv(input_,32)
  conv1_2=basic_conv(conv1_1,32)
  pool1_1=MaxPooling2D()(conv1_2)
  pool1_2=AveragePooling2D()(conv1_2)
  pool1=concatenate([pool1_1,pool1_2])
  
  conv2_1=basic_conv(pool1,32)
  conv2_2=basic_conv(conv2_1,32)
  pool2_1=MaxPooling2D()(conv2_2)
  pool2_2=AveragePooling2D()(conv2_2)
  pool2=concatenate([pool2_1,pool2_2])
  
  conv3_1=basic_conv(pool2,64)
  conv3_2=basic_conv(conv3_1,64)
  conv3_3=basic_conv(conv3_2,64)
  pool3_1=MaxPooling2D()(conv3_3)
  pool3_2=AveragePooling2D()(conv3_3)
  pool3=concatenate([pool3_1,pool3_2])
  
  conv4_1=basic_conv(pool3,64)
  conv4_2=basic_conv(conv4_1,64)
  conv4_3=basic_conv(conv4_2,64)
  pool4_1=MaxPooling2D()(conv4_3)
  pool4_2=AveragePooling2D()(conv4_3)
  pool4=concatenate([pool4_1,pool4_2])
  
  conv5_1=basic_conv(pool4,64)
  conv5_2=basic_conv(conv5_1,64)
  conv5_3=basic_conv(conv5_2,64)
  pool5_1=MaxPooling2D()(conv5_3)
  pool5_2=AveragePooling2D()(conv5_3)
  pool5=concatenate([pool5_1,pool5_2])
  
  flatten=Flatten()(pool5)
  
  dense1=Dense(512,activation="relu")(flatten)
  dense2=Dense(256,activation="relu")(dense1)
  dense3=Dense(2,activation="sigmoid")(dense2)
  
  model=Model(input=input_,output=dense3)
  return model
def VGG16_3(CLF_SIZE):
  input_=Input(shape=(CLF_SIZE,CLF_SIZE,1))
  
  conv1_1=basic_conv(input_,32)
  conv1_2=basic_conv(conv1_1,32)
  conv1_2=concatenate([conv1_1,conv1_2])
  pool1_1=MaxPooling2D()(conv1_2)
  pool1_2=AveragePooling2D()(conv1_2)
  pool1=concatenate([pool1_1,pool1_2])
  
  conv2_1=basic_conv(pool1,32)
  conv2_2=basic_conv(conv2_1,32)
  conv2_2=concatenate([conv2_1,conv2_2])
  pool2_1=MaxPooling2D()(conv2_2)
  pool2_2=AveragePooling2D()(conv2_2)
  pool2=concatenate([pool2_1,pool2_2])
  
  conv3_1=basic_conv(pool2,64)
  conv3_2=basic_conv(conv3_1,64)
  conv3_3=basic_conv(conv3_2,64)
  conv3_3=concatenate([conv3_1,conv3_2,conv3_3])
  pool3_1=MaxPooling2D()(conv3_3)
  pool3_2=AveragePooling2D()(conv3_3)
  pool3=concatenate([pool3_1,pool3_2])
  
  conv4_1=basic_conv(pool3,64)
  conv4_2=basic_conv(conv4_1,64)
  conv4_3=basic_conv(conv4_2,64)
  conv4_3=concatenate([conv4_1,conv4_2,conv4_3])
  pool4_1=MaxPooling2D()(conv4_3)
  pool4_2=AveragePooling2D()(conv4_3)
  pool4=concatenate([pool4_1,pool4_2])
  
  conv5_1=basic_conv(pool4,64)
  conv5_2=basic_conv(conv5_1,64)
  conv5_3=basic_conv(conv5_2,64)
  conv5_3=concatenate([conv5_1,conv5_2,conv5_3])
  pool5_1=MaxPooling2D()(conv5_3)
  pool5_2=AveragePooling2D()(conv5_3)
  pool5=concatenate([pool5_1,pool5_2])
  
  flatten=Flatten()(pool5)
  
  
  dense1=Dense(512,activation="relu")(flatten)
  dense2=Dense(256,activation="relu")(dense1)
  dense3=Dense(2,activation="sigmoid")(dense2)
  
  model=Model(input=input_,output=dense3)
  return model
def DENSE_VGG16(CLF_SIZE):
  input_=Input(shape=(CLF_SIZE,CLF_SIZE,1))
  
  conv1_1=basic_conv(input_,16)
  conv1_1=concatenate([input_,conv1_1])
  conv1_2=basic_conv(conv1_1,16)
  conv1_2=concatenate([conv1_1,conv1_2])
  pool1=MaxPooling2D()(conv1_2)
  
  conv2_1=basic_conv(pool1,16)
  conv2_1=concatenate([pool1,conv2_1])
  conv2_2=basic_conv(conv2_1,16)
  conv2_2=concatenate([conv2_1,conv2_2])
  pool2=MaxPooling2D()(conv2_2)
  
  conv3_1=basic_conv(pool2,32)
  conv3_1=concatenate([pool2,conv3_1])
  conv3_2=basic_conv(conv3_1,32)
  conv3_2=concatenate([conv3_1,conv3_2])
  conv3_3=basic_conv(conv3_2,32)
  conv3_3=concatenate([conv3_2,conv3_3])
  pool3=MaxPooling2D()(conv3_3)
  
  conv4_1=basic_conv(pool3,32)
  conv4_1=concatenate([pool3,conv4_1])
  conv4_2=basic_conv(conv4_1,32)
  conv4_2=concatenate([conv4_1,conv4_2])
  conv4_3=basic_conv(conv4_2,32)
  conv4_3=concatenate([conv4_2,conv4_3])
  pool4=MaxPooling2D()(conv4_3)
  
  conv5_1=basic_conv(pool4,32)
  conv5_1=concatenate([pool4,conv5_1])
  conv5_2=basic_conv(conv5_1,32)
  conv5_2=concatenate([conv5_1,conv5_2])
  conv5_3=basic_conv(conv5_2,32)
  conv5_3=concatenate([conv5_2,conv5_3])
  pool5=MaxPooling2D()(conv5_3)
  
  flatten=Flatten()(pool5)
  
  dense1=Dense(512,activation="relu")(flatten)
  dense2=Dense(256,activation="relu")(dense1)
  dense3=Dense(2,activation="sigmoid")(dense2)
  
  model=Model(input=input_,output=dense3)
  return model

def DENSE_VGG16_2(CLF_SIZE):
  input_=Input(shape=(CLF_SIZE,CLF_SIZE,1))
  
  conv1_1=basic_conv(input_,32)
  conv1_1=concatenate([input_,conv1_1])
  conv1_2=basic_conv(conv1_1,32)
  conv1_2=concatenate([conv1_1,conv1_2])
  pool1=MaxPooling2D()(conv1_2)
  
  conv2_1=basic_conv(pool1,32)
  conv2_1=concatenate([pool1,conv2_1])
  conv2_2=basic_conv(conv2_1,32)
  conv2_2=concatenate([conv2_1,conv2_2])
  pool2=MaxPooling2D()(conv2_2)
  
  conv3_1=basic_conv(pool2,64)
  conv3_1=concatenate([pool2,conv3_1])
  conv3_2=basic_conv(conv3_1,64)
  conv3_2=concatenate([conv3_1,conv3_2])
  conv3_3=basic_conv(conv3_2,64)
  conv3_3=concatenate([conv3_2,conv3_3])
  pool3=MaxPooling2D()(conv3_3)
  
  conv4_1=basic_conv(pool3,64)
  conv4_1=concatenate([pool3,conv4_1])
  conv4_2=basic_conv(conv4_1,64)
  conv4_2=concatenate([conv4_1,conv4_2])
  conv4_3=basic_conv(conv4_2,64)
  conv4_3=concatenate([conv4_2,conv4_3])
  pool4=MaxPooling2D()(conv4_3)
  
  conv5_1=basic_conv(pool4,64)
  conv5_1=concatenate([pool4,conv5_1])
  conv5_2=basic_conv(conv5_1,64)
  conv5_2=concatenate([conv5_1,conv5_2])
  conv5_3=basic_conv(conv5_2,64)
  conv5_3=concatenate([conv5_2,conv5_3])
  pool5=MaxPooling2D()(conv5_3)
  
  flatten=Flatten()(pool5)
  
  dense1=Dense(512,activation="relu")(flatten)
  dense2=Dense(256,activation="relu")(dense1)
  dense3=Dense(2,activation="sigmoid")(dense2)
  
  model=Model(input=input_,output=dense3)
  return model




def identity_block(X, f, filters, stage, block):
    """
    Implementation of the identity block as defined in Figure 3
    
    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network
    
    Returns:
    X -- output of the identity block, tensor of shape (n_H, n_W, n_C)
    """
    
    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    # Retrieve Filters
    F1, F2, F3 = filters
    
    # Save the input value. You'll need this later to add back to the main path. 
    X_shortcut = X
    
    # First component of main path
    X = Conv2D(filters = F1, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2a', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)
    
    ### START CODE HERE ###
    
    # Second component of main path (≈3 lines)
    X = Conv2D(filters=F2, kernel_size =(f, f), strides =(1,1), padding='same', name=conv_name_base+'2b', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base+'2b')(X)
    X = Activation('relu')(X)

    # Third component of main path (≈2 lines)
    X = Conv2D(filters=F3, kernel_size=(1,1), strides=(1,1), padding="valid", name=conv_name_base+'2c', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base+'2c')(X)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)
    
    ### END CODE HERE ###
    
    return X

def convolutional_block(X, f, filters, stage, block, s = 2):
    """
    Implementation of the convolutional block as defined in Figure 4
    
    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network
    s -- Integer, specifying the stride to be used
    
    Returns:
    X -- output of the convolutional block, tensor of shape (n_H, n_W, n_C)
    """
    
    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    # Retrieve Filters
    F1, F2, F3 = filters
    
    # Save the input value
    X_shortcut = X


    ##### MAIN PATH #####
    # First component of main path 
    X = Conv2D(F1, (1, 1), strides = (s,s), name = conv_name_base + '2a', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)
    
    ### START CODE HERE ###

    # Second component of main path (≈3 lines)
    X = Conv2D(F2, kernel_size=(f,f), strides=(1,1), padding="same",name=conv_name_base+'2b', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base+'2b')(X)
    X = Activation('relu')(X)

    # Third component of main path (≈2 lines)
    X = Conv2D(F3, kernel_size=(1,1), strides=(1,1), padding="valid", name=conv_name_base+'2c', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base+'2c')(X)


    ##### SHORTCUT PATH #### (≈2 lines)
    X_shortcut = Conv2D(F3, kernel_size=(1,1), strides=(s,s), padding="valid",name=conv_name_base+'1', kernel_initializer=glorot_uniform(seed=0))(X_shortcut)
    X_shortcut = BatchNormalization(axis=3, name=bn_name_base+'1')(X_shortcut)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)
    
    ### END CODE HERE ###
    
    return X

def ResNet50Network(X_input, classes=6) :
    # Zero-Padding
    X = ZeroPadding2D((3, 3))(X_input)
    
    # Stage 1
    X = Conv2D(64, (7, 7), strides = (2, 2), name = 'conv1', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = 'bn_conv1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)

    # Stage 2
    X = convolutional_block(X, f = 3, filters = [64, 64, 256], stage = 2, block='a', s = 1)
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='b')
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='c')

    ### START CODE HERE ###
    # Stage 3 (≈4 lines)
    X = convolutional_block(X, f=3, filters=[128,128,512], s=2, stage=3, block='a' )
    for _block in ['b','c','d'] :
        X = identity_block(X,f=3, filters=[128,128,512], stage=3, block=_block)
    

    # Stage 4 (≈6 lines)
    X = convolutional_block(X, f=3, filters=[256,256,1024], s=2, block='a', stage=4)
    for _block in ['b','c','d','e','f']:
        X = identity_block(X, f=3, filters=[256,256,1024], stage=4, block=_block)




    # Stage 5 (≈3 lines)
    X = convolutional_block(X, f=3, filters=[512,512,2048], s=2, block='a', stage=5)
    for block_name in ['b','c'] :
        X = identity_block(X, f=3, filters=[512,512,2048], stage=5, block=block_name)
    
    # AVGPOOL (≈1 line). Use "X = AveragePooling2D(...)(X)"
    X = AveragePooling2D((2, 2), name='avg_pool')(X)
    
    ### END CODE HERE ###

    # output layer
    X = Flatten()(X)
    X = Dense(classes, activation='sigmoid', name='fc' + str(classes), kernel_initializer = glorot_uniform(seed=0))(X)

    return X
 

def ResNet50(CLF_SIZE):
    """
    Implementation of the popular ResNet50 the following architecture:
    CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3
    -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> TOPLAYER
    Arguments:
    input_shape -- shape of the images of the dataset
    classes -- integer, number of classes
    Returns:
    model -- a Model() instance in Keras
    """
    input_shape = (CLF_SIZE, CLF_SIZE, 1)
    classes = 2
    # Define the input as a tensor with shape input_shape
    X_input = Input(input_shape,name="layer_x")
    X = ResNet50Network(X_input,classes)
    
    # Create model
    model = Model(inputs = X_input, outputs = X, name='ResNet50')

    return model




#model = ResNet50(input_shape = (64, 64, 3), classes = 6)
#model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
if __name__=="__main__":
  vgg=ResNet50(256)
  vgg.summary()
import tensorflow as tf
from tensorflow.keras.layers import Conv2D,MaxPooling2D,BatchNormalization,Activation,Conv2DTranspose,Dropout
from tensorflow.keras import Model

import os
import importlib
from tensorflow.keras.layers import concatenate, add
import numpy as np


def conv2d_block(input_tensor, n_filters, kernel_size=3,strides=(2,2),batchnorm=True):
    """
    Function to add convolutional layers with the parameters 
    """
    
    #first layer
    x=Conv2D(filters=n_filters,kernel_size=(kernel_size,kernel_size),strides=strides,
            kernel_initializer='he_normal',padding='same')(input_tensor)
    if batchnorm:
        x=BatchNormalization()(x)
    x=Activation('relu')(x)
    
    #second layer
    #TB Implemented
    
    return x
    
   
   
def encoder(input_tile,n_filters,kernel_size=3,strides=(2,2),max_pool_shape=(2,2),dropout=0.1,maxpool=True,batchnorm=True):
    c=conv2d_block(input_tile,n_filters,kernel_size=kernel_size,strides=strides,batchnorm=batchnorm)
    if maxpool:
        p=MaxPooling2D(max_pool_shape)(c) 
        p=Dropout(dropout)(p)
    else:
        p=Dropout(dropout)(c)
    return p,c
 

   
def decoder(inp,to_conc,n_filters=None,kernel_size=(2,2),strides=(2,2),padding='valid',dropout=0.1):
    #print("inp.shape: ",inp.shape)
    u=Conv2DTranspose(n_filters,kernel_size=kernel_size,strides=strides,padding=padding)(inp)
    print("before concat: (deconv.shape,to_conc.shape) : ",u.shape,to_conc.shape)
    u=concatenate([u,to_conc])
    print("after concat: deconv.shape  ",u.shape)
    u=Dropout(dropout)(u)
    return u
    
    
    
def unet(input_tile,n_filters=16,dropout=0.1,maxpool=True,batchnorm=True):
    #Encoding Path
    #contracting path
    if np.ndim(input_tile.shape)==2:
        input_tile=input_tile.reshape(input_tile.shape[0],input_tile.shape[1],1)
    p1,c1=encoder(input_tile,n_filters*1,kernel_size=2,strides=(2,2),maxpool=maxpool,max_pool_shape=(2,2),batchnorm=batchnorm)     
    print(p1.shape,c1.shape)
    p2,c2=encoder(p1,n_filters*2,kernel_size=2,strides=(2,2),maxpool=maxpool,max_pool_shape=(2,2),batchnorm=batchnorm)
    print(p2.shape,c2.shape)
    p3,c3=encoder(p2,n_filters*4,kernel_size=2,strides=(2,2),maxpool=maxpool,max_pool_shape=(2,2),batchnorm=batchnorm)
    print(p3.shape,c3.shape)
    p4,c4=encoder(p3,n_filters*8,kernel_size=2,strides=(2,2),maxpool=maxpool,max_pool_shape=(2,2),batchnorm=batchnorm)
    print(p4.shape,c4.shape)
    p5,c5=encoder(p4,n_filters*16,kernel_size=2,strides=(2,2),maxpool=maxpool,max_pool_shape=(2,2),batchnorm=batchnorm)
    print(p5.shape,c5.shape)
    p6,c6=encoder(p5,n_filters*32,kernel_size=2,strides=(2,2),maxpool=maxpool,max_pool_shape=(2,2),batchnorm=batchnorm)
    print(p6.shape,c6.shape)
    print("ENCDEC")
    #expanding path
    
    u7=decoder(c6,c5,n_filters=n_filters*16,kernel_size=(2,2),strides=(2,2),padding='valid')
    print(u7.shape)
    u8=decoder(u7,c4,n_filters=n_filters*8,kernel_size=(2,2),strides=(2,2),padding='valid')
    print(u8.shape)
    u9=decoder(u8,c3,n_filters=n_filters*4,kernel_size=(2,2),strides=(2,2),padding='valid')
    print(u9.shape)
    u10=decoder(u9,c2,n_filters=n_filters*2,kernel_size=(2,2),strides=(2,2),padding='valid')
    print(u10.shape)
    u11=decoder(u10,c1,n_filters=n_filters*1,kernel_size=(2,2),strides=(2,2),padding='valid')
    print(u11.shape)
    u12=Conv2DTranspose(1,kernel_size=(2,2),strides=(2,2),padding="valid")(u11)
    #print(input_tile.shape)
    #dum=tf.constant(np.zeros((None,256,256,1)))
    #u12=concatenate([u12,dum])
    #u12=decoder(u11,input_tile,n_filters=n_filters,kernel_size=(2,2),strides=(2,2),padding='valid')
    print(u12.shape)
    #u13=Conv2DTranspose(1,kernel_size=(2,1),strides=(2,1),padding="valid")(u12)
    #print(u13.shape)
   
    print("final: ",u12.shape)
    model=Model(inputs=[input_tile], outputs=[u12])
    return model
    
    
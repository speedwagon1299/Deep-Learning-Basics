import tensorflow as tf
import numpy as np
import scipy.misc
from tensorflow.keras.applications.resnet_v2 import ResNet50V2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet_v2 import preprocess_input, decode_predictions
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from tensorflow.keras.models import Model, load_model
from resnets_utils import *
from tensorflow.keras.initializers import random_uniform, glorot_uniform, constant, identity
from tensorflow.python.framework.ops import EagerTensor
from matplotlib.pyplot import imshow

np.random.seed(1)
tf.random.set_seed(2)


def identity_block(X, f, filters, training=True, initializer=random_uniform):
    """    
    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    training -- True: Behave in training mode
                False: Behave in inference mode
    initializer -- to set up the initial weights of a layer. Equals to random uniform initializer
    Returns:
    X -- output of the identity block, tensor of shape (m, n_H, n_W, n_C)
    """
    
    F1, F2, F3 = filters
    
    X_shortcut = X
    
    
    X = Conv2D(filters = F1,kernel_size = (1,1),strides = (1,1),padding= 'valid',kernel_initializer=initializer(seed = 0))(X)
    X = BatchNormalization(axis = 3)(X, training = training)
    X = Activation('relu')(X)
    
    X = Conv2D(filters = F2,kernel_size = (f,f),strides = (1,1),padding='same',kernel_initializer=initializer(seed = 0))(X)
    X = BatchNormalization(axis = 3)(X, training = training)
    X = Activation('relu')(X)

    
    X = Conv2D(filters = F3,kernel_size = (1,1),strides = (1,1),padding ='valid',kernel_initializer=initializer(seed = 0))(X)
    X = BatchNormalization(axis = 3)(X, training = training)
    
    X = Add()([X,X_shortcut])
    X = Activation('relu')(X)

    return X



def convolutional_block(X, f, filters, s = 2, training=True, initializer=glorot_uniform):
    """
    
    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    s -- Integer, specifying the stride to be used
    training -- True: Behave in training mode
                False: Behave in inference mode
    initializer -- to set up the initial weights of a layer. Equals to Glorot uniform initializer, 
                   also called Xavier uniform initializer.
    
    Returns:
    X -- output of the convolutional block, tensor of shape (m, n_H, n_W, n_C)
    """
    
    F1, F2, F3 = filters
    X_shortcut = X

    
    X = Conv2D(filters = F1, kernel_size = 1, strides = (s, s), padding='valid', kernel_initializer = initializer(seed=0))(X)
    X = BatchNormalization(axis = 3)(X, training=training)
    X = Activation('relu')(X)


    
    X = Conv2D(filters = F2, kernel_size = (f,f), strides = (1,1), padding='same', kernel_initializer = initializer(seed=0))(X)
    X = BatchNormalization(axis = 3)(X, training=training) 
    X = Activation('relu')(X)

    X = Conv2D(filters = F3, kernel_size = (1,1), strides = (1,1), padding='valid', kernel_initializer = initializer(seed=0))(X) 
    X = BatchNormalization(axis = 3)(X, training=training) 
    
    X_shortcut = Conv2D(filters = F3, kernel_size = (1,1), strides = (s,s), padding='valid', kernel_initializer = initializer(seed=0))(X_shortcut)
    X_shortcut = BatchNormalization(axis = 3)(X_shortcut, training=training)
    

    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)
    
    return X



def ResNet50(input_shape = (64, 64, 3), classes = 6):
    """
    Stage-wise implementation of the architecture of the popular ResNet50:
    CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3
    -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> FLATTEN -> DENSE 

    Arguments:
    input_shape -- shape of the images of the dataset
    classes -- integer, number of classes

    Returns:
    model -- a Model() instance in Keras
    """
    
    X_input = Input(input_shape)

    X = ZeroPadding2D((3, 3))(X_input)
    
    X = Conv2D(64, (7, 7), strides = (2, 2), kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3)(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)

    X = convolutional_block(X, f = 3, filters = [64, 64, 256], s = 1)
    X = identity_block(X, 3, [64, 64, 256])
    X = identity_block(X, 3, [64, 64, 256])

    
    X = convolutional_block(X,f=3,filters = [128,128,512],s = 2)
    X = identity_block(X,f=3,filters = [128,128,512])
    X = identity_block(X,f=3,filters = [128,128,512])
    X = identity_block(X,f=3,filters = [128,128,512])

    X = convolutional_block(X,f=3,filters=[256,256,1024],s=2)
    
    X = identity_block(X,f=3,filters = [256,256,1024])
    X = identity_block(X,f=3,filters = [256,256,1024])
    X = identity_block(X,f=3,filters = [256,256,1024])
    X = identity_block(X,f=3,filters = [256,256,1024])
    X = identity_block(X,f=3,filters = [256,256,1024])

    X = convolutional_block(X,f=3,filters=[512,512,2048],s=2)
    
    X = identity_block(X,f=3,filters =[512,512,2048])
    X = identity_block(X,f=3,filters =[512,512,2048])

    X = AveragePooling2D()(X)
    
    X = Flatten()(X)
    X = Dense(classes, activation='softmax', kernel_initializer = glorot_uniform(seed=0))(X)
    
    model = Model(inputs = X_input, outputs = X)

    return model


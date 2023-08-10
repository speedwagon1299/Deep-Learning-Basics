import math
import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as tfl
from tensorflow.python.framework import ops

def happyModel():
    """
    Implements the forward propagation for the binary classification model:
    ZEROPAD2D -> CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> FLATTEN -> DENSE
    Sequential model kinda weird ngl
    """
    model = tf.keras.Sequential([
            ## ZeroPadding2D with padding 3, input shape of 64 x 64 x 3
            tfl.ZeroPadding2D(padding=(3, 3), input_shape=(64, 64, 3)),
           
            ## Conv2D with 32 7x7 filters and stride of 1
            tfl.Conv2D(32, (7,7)),
            
            ## BatchNormalization for axis 3
            tfl.BatchNormalization(axis=-1),
            
            ## ReLU
            tfl.ReLU(),
           
            ## Max Pooling 2D with default parameters
            tfl.MaxPool2D(),
           
            ## Flatten layer
            tfl.Flatten(),
           
            ## Dense layer with 1 unit for output & 'sigmoid' activation
            tfl.Dense(1, activation='sigmoid')
        ])
    # V IMP!!! : every layer allows param input_shape = (tuple), use it only on the first layer
    return model

# GRADED FUNCTION: convolutional_model

def convolutional_model(input_shape):
    """
    Implements the forward propagation for the model:
    CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> DENSE
    Functional is wayyy better, keep the new input in brackets after the function
    """

    input_img = tf.keras.Input(shape=input_shape)
    
    ## CONV2D: 8 filters 4x4, stride of 1, padding 'SAME'
    Z1 = tfl.Conv2D(filters = 8, kernel_size = (4,4), padding = 'same', strides = 1)(input_img)
    
    ## RELU
    A1 = tfl.ReLU()(Z1)
    
    ## MAXPOOL: window 8x8, stride 8, padding 'SAME'
    P1 = tfl.MaxPooling2D(pool_size = (8,8), strides = 8, padding = 'same') (A1) 
    
    ## CONV2D: 16 filters 2x2, stride 1, padding 'SAME'
    Z2 = tfl.Conv2D(16,(2,2),1,'same')(P1)
    
    ## RELU
    A2 = tfl.ReLU()(Z2)
    
    ## MAXPOOL: window 4x4, stride 4, padding 'SAME'
    P2 = tfl.MaxPooling2D((4,4),4,'same')(A2)
    
    ## FLATTEN
    F = tfl.Flatten()(P2)
    
    ## Dense layer
    ## 6 neurons in output layer. Hint: one of the arguments should be "activation='softmax'" 
    outputs = tfl.Dense(units = 6, activation = 'softmax')(F)

    model = tf.keras.Model(inputs=input_img, outputs=outputs)
    return model


# Good Docs :
# https://www.tensorflow.org/guide/keras/sequential_model
# https://www.tensorflow.org/guide/keras/functional_api

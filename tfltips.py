import math
import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as tfl
from tensorflow.python.framework import ops

def happyModel():
    """
    Implements the forward propagation for the binary classification model:
    ZEROPAD2D -> CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> FLATTEN -> DENSE
   
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

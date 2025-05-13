from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import dill, random
import itertools
import sys
import logging


import keras
import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tensorflow.python.platform import flags
from tensorflow.python.platform import flags
from keras import Input
from keras import backend as K
from keras.models import Sequential, load_model
from keras.layers.convolutional import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers.core import Activation, Flatten, Dense, Dropout
from keras.layers.advanced_activations import ELU
from keras.utils import np_utils
# from keras.optimizers import Adam, SGD, Nadam, Adamax, RMSprop
from keras.callbacks import TensorBoard
from tensorflow.keras.layers import BatchNormalization
tf.set_random_seed(1234)
Set = set()
rng = np.random.RandomState([2017, 8, 30])
tf.logging.set_verbosity(tf.logging.ERROR )

class ConvNet:
    @staticmethod

    def build(input_shape, classes):
        model = Sequential()
        #Block1
        model.add(Conv2D(32, kernel_size=(1, 8), padding="same", input_shape=input_shape, name='block1_conv1'))
        model.add(ELU(alpha=1.0, name='block1_act1'))
        model.add(Conv2D(32, kernel_size=(1, 8), padding="same", name='block1_conv2'))
        model.add(ELU(alpha=1.0, name='block1_act2'))
        model.add(MaxPooling2D(pool_size=(1, 8), strides=(1, 4), padding="same", name='block1_pool'))
        model.add(Dropout(0.1))

        # Block2
        model.add(Conv2D(64, kernel_size=(1, 8), padding="same", name='block2_conv1'))
        model.add(ELU(alpha=1.0, name='block2_act1'))
        model.add(Conv2D(64, kernel_size=(1, 8), padding="same", name='block2_conv2'))
        model.add(ELU(alpha=1.0, name='block2_act2'))
        model.add(MaxPooling2D(pool_size=(1, 8), strides=(1, 4), padding="same", name='block2_pool'))
        model.add(Dropout(0.1))

        # Block3
        model.add(Conv2D(128, kernel_size=(1, 8), padding="same", name='block3_conv1'))
        model.add(ELU(alpha=1.0, name='block3_act1'))
        model.add(Conv2D(128, kernel_size=(1, 8), padding="same", name='block3_conv2'))
        model.add(ELU(alpha=1.0, name='block3_act2'))
        model.add(MaxPooling2D(pool_size=(1, 8), strides=(1, 4), padding="same", name='block3_pool'))
        model.add(Dropout(0.1))

        # Block4
        model.add(Conv2D(128, kernel_size=(1, 8), padding="same", name='block4_conv1'))
        model.add(ELU(alpha=1.0, name='block4_act1'))
        model.add(Conv2D(128, kernel_size=(1, 8), padding="same", name='block4_conv2'))
        model.add(ELU(alpha=1.0, name='block4_act2'))
        model.add(MaxPooling2D(pool_size=(1, 8), strides=(1, 4), padding="same", name='block4_pool'))
        model.add(Dropout(0.1))


        model.add(Flatten(name='flatten'))
        model.add(Dense(512, activation='relu', name='fc1'))
        model.add(Dropout(0.3))
        model.add(Dense(512, activation='relu', name='fc2'))
        model.add(Dropout(0.3))
        model.add(Dense(classes, activation='softmax', name='prediction'))
        
        return model
    
class OurConvNet:
    @staticmethod
    def build(input_shape, classes):
        model = Sequential()
        #Block1
        model.add(Conv2D(32, kernel_size=(1, 8),strides = (1,4),  padding="same", input_shape=input_shape, name='block1_conv1'))
        model.add(ELU(alpha=1.0, name='block1_act1'))
        
        model.add(Conv2D(64, kernel_size=(1, 8),strides = (1,4), padding="same", name='block1_conv2'))
        model.add(ELU(alpha=1.0, name='block1_act2'))
        
        model.add(Conv2D(128, kernel_size=(1, 8),strides = (1,4), padding="same", name='block1_conv3'))
        model.add(ELU(alpha=1.0, name='block1_act3'))
        
        
        model.add(Flatten(name='flatten'))
        model.add(Dense(512, activation='relu', name='fc1'))
        model.add(Dense(classes, activation='softmax', name='prediction'))
        
        return model

class AWFConvNet:
    @staticmethod
    def build(input_shape, classes):
        model = Sequential()
        #Block1
        model.add(Dropout(input_shape=input_shape, rate=0.1))
        
        model.add(Conv2D(32, kernel_size=(1, 5),strides = (1,1),  padding='valid', activation='relu'))
        model.add(MaxPooling2D(pool_size=(1, 4), strides=(1, 1), padding='valid'))

        
        model.add(Conv2D(32, kernel_size=(1, 5),strides = (1,1),  padding='valid', activation='relu'))
        model.add(MaxPooling2D(pool_size=(1, 4), strides=(1, 1), padding='valid'))                        
        
        model.add(Flatten(name='flatten'))
        model.add(Dense(classes, activation='softmax', name='prediction'))
        
        return model    

def scale(X):
    # scale the data
    Xmin = abs(X.min(axis=0))
    Xmax = abs(X.max(axis=0))
    Xscale = (np.max(np.vstack((abs(Xmin), abs(Xmax))), axis=0)).astype(np.float32)
    X = X / Xscale
    return X

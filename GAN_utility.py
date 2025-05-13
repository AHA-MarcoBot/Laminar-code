import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.keras.layers import Input, Conv1D, Activation, Lambda, Conv2DTranspose, \
    LeakyReLU, Dropout, Flatten, ELU, MaxPooling1D, Reshape, Average, ReLU, Lambda, Dense
from tensorflow.keras.layers import BatchNormalization


def generator_model_5():
    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Dense(512))
    # model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))
    model.add(Dropout(0.05))

    model.add(tf.keras.layers.Dense(512))
    # model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))
    model.add(Dropout(0.05))

    model.add(tf.keras.layers.Dense(1024))
    # model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))
    model.add(Dropout(0.05))

    model.add(tf.keras.layers.Dense(1024))
    # model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))
    model.add(Dropout(0.05))

    model.add(tf.keras.layers.Dense(2000))
    # model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))
    model.add(Dropout(0.05))

    model.add(tf.keras.layers.Dense(2000))
    model.add(tf.keras.layers.Activation('relu'))
    return model

def discriminator_model_5():
    model = tf.keras.Sequential()

    model.add(Dense(2000, input_shape=(2000,)))  
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.05))

    model.add(Dense(1024))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.05))

    model.add(Dense(512))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.05))

    model.add(Dense(256))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.05))

    model.add(Dense(1, activation='sigmoid'))  # 输出真假概率

    return model

def patch_model_10():
    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Dense(512))
    # model.add(BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))
    model.add(Dropout(0.05))

    model.add(tf.keras.layers.Dense(512))
    # model.add(BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))
    model.add(Dropout(0.05))

    model.add(tf.keras.layers.Dense(1024))
    # model.add(BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))
    model.add(Dropout(0.05))

    model.add(tf.keras.layers.Dense(1024))
    # model.add(BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))
    model.add(Dropout(0.05))

    model.add(tf.keras.layers.Dense(1024))
    # model.add(BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))
    model.add(Dropout(0.05))

    model.add(tf.keras.layers.Dense(10))
    model.add(tf.keras.layers.Activation('relu'))
    return model
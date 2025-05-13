#####测试Laminar模仿为目标类别的能力
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"
import pickle
import dill
import tensorflow as tf
import mockingbird_utility as mb_utility
from keras.optimizers import adam_v2, adamax_v2
from keras import models
import csv
from keras.utils import np_utils

irrelevent_test_ratio = 1.0

def get_class_samples(X, Y, C):
    y = np.argmax(Y, axis=1)
    ind = np.where(y == C)
    return X[ind], Y[ind]

data_dir="advdata/"
train_data_path = data_dir + "train.dill"
train_data, train_label = dill.load(open(train_data_path,"rb"),encoding="bytes")
train_data = np.array(train_data)
train_data = np.expand_dims(train_data, 1)
train_label = np.array(train_label)
print("Train data: ", train_data.shape, train_label.shape)

def load_number_pairs(file_path):
    with open(file_path, "r") as file:
        reader = csv.reader(file)
        next(reader)
        return {int(row[0]): int(row[1]) for row in reader}

num_bursts=2000
num_classes=200
learning_rate=0.002
VALIDATION_SPLIT= 0.1
NB_EPOCH = 20
pairs_path = "./pairs/pairs.csv"
pairs_dict = load_number_pairs(pairs_path)

input_shape=(1, num_bursts, 1)

model_save_path = "./model/origin_AWFdata_DF_epoch=30_model.h5"
model = models.load_model(model_save_path)
model.trainable = False

true_labels = np.argmax(train_label, axis=-1)
target_labels = np.ones_like(true_labels) * 199

prediction = model.evaluate(train_data, np_utils.to_categorical(true_labels, num_classes))
print(prediction)

prediction = model.evaluate(train_data, np_utils.to_categorical(target_labels, num_classes))
print(prediction)
   

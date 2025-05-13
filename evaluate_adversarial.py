import numpy as np
import matplotlib.pyplot as plt
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"
import pickle
import dill
import tensorflow as tf
import mockingbird_utility as mb_utility
from keras.optimizers import adam_v2, adamax_v2
import random
from tensorflow.keras.optimizers import Adam, RMSprop, SGD, Adagrad
from keras.utils import np_utils
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, precision_score


def get_class_samples(X, Y, C):
    y = np.argmax(Y, axis=1)
    ind = np.where(y == C)
    return X[ind], Y[ind]

def load_data(path, maxlen=None, minlen=0, traces=0, dnn_type=None, openw=False):
    """Load and shape the dataset"""
    npzfile = np.load(path, allow_pickle=True)
    data = npzfile["data"]
    labels = npzfile["labels"]
    npzfile.close()
    print("data shape", data.shape, labels.shape)
    return data, labels

def ext_dim(onehot):
    return np.concatenate([onehot, np.zeros((onehot.shape[0], conc))], axis=1)

data_dir="advdata/"
start = 0

train_data_path = data_dir + "user1#1/train.dill"
train_data, train_label = dill.load(open(train_data_path,"rb"),encoding="bytes")
train_data = np.array(train_data)
train_data = np.expand_dims(train_data, 1)
train_label = np.array(train_label)
print("Train data: ", train_data.shape, train_label.shape)

test1_data_path = data_dir + "user1#1/test.dill"
test1_data, test1_label = dill.load(open(test1_data_path,"rb"),encoding="bytes")
test1_data = np.array(test1_data)
test1_data = np.expand_dims(test1_data, 1)
test1_label = np.array(test1_label)

test2_data_path = data_dir + "user2#1/test.dill"
test2_data, test2_label = dill.load(open(test2_data_path,"rb"),encoding="bytes")
test2_data = np.array(test2_data)
test2_data = np.expand_dims(test2_data , 1)
test2_label = np.array(test2_label)

num_bursts=2000
num_classes=200
learning_rate=0.002
VALIDATION_SPLIT= 0.1

NB_EPOCH = 20

input_shape=(1, num_bursts, 1)

old_train_data, old_train_label = load_data("./dataset/Burst_Closed World/burst_tor_200w_2500tr_test.npz")
old_train_data = old_train_data.reshape((old_train_data.shape[0], 1, old_train_data.shape[2], 1))
old_train_data = old_train_data.astype("float32")

start = 0
conc = 5
result = []
prec = []
falsepos = []
for label in range(start, start+30-conc+1, conc):
    targetLabel = 199
    
    new_train_data = old_train_data
    new_train_label = ext_dim(old_train_label)
    for i in range(0, conc):
        data_item, _ = get_class_samples(train_data, train_label, label+i)
        new_train_data = np.concatenate([new_train_data, data_item], axis=0)
        label_item = np_utils.to_categorical(np.full((data_item.shape[0], ), 200+i), num_classes=num_classes+conc)
        new_train_label = np.concatenate([new_train_label, label_item], axis=0)

    BATCH_SIZE = 128
    OPTIMIZER = adamax_v2.Adamax(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model = mb_utility.ConvNet.build(input_shape=input_shape, classes=num_classes+conc)

    model.compile(loss="categorical_crossentropy", optimizer=OPTIMIZER,metrics=["acc"])

    print("new_data: ", new_train_data.shape)

    indices = np.arange(new_train_data.shape[0])
    np.random.shuffle(indices)
    new_train_data = new_train_data[indices]
    new_train_label = new_train_label[indices]

    history = model.fit(new_train_data, new_train_label, batch_size=128, epochs=NB_EPOCH, validation_split=VALIDATION_SPLIT)

    for i in range(0, conc):
        single_test_data, single_test_label = get_class_samples(test2_data, test2_label, label+i)
        single_test_label = np_utils.to_categorical(np.full((single_test_data.shape[0], ), 200+i), num_classes=num_classes+conc)
        val = model.evaluate(single_test_data, single_test_label)

        label_pred_probs = model.predict(single_test_data)
        label_pred = np.argmax(label_pred_probs, axis=1)
        label_pred = np_utils.to_categorical(label_pred, num_classes = num_classes+conc)

        print(f"class {label+i} accuracy:", val[1])
        result.append(val[1])

print("Average Accuracy: ", np.nanmean(result))

plt.figure()
plt.plot(result, label="Accuracy", linestyle='-', color='red', marker='s', markerfacecolor='red', markersize=3, linewidth=1)
plt.title("Adversarial Training Accuracy")
plt.xlabel("label")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig("./adversarial.png")
plt.close()
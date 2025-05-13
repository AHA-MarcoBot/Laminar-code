import numpy as np
import matplotlib.pyplot as plt
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"
import pickle
import dill
import random
from keras.utils import np_utils

def get_class_samples(X, Y, C):
    ind = np.where(Y == C)
    return X[ind], Y[ind]

def load_data(path, maxlen=None, minlen=0, traces=0, dnn_type=None, openw=False):
    """Load and shape the dataset"""
    npzfile = np.load(path, allow_pickle=True)
    data = npzfile["data"]
    labels = npzfile["labels"]

    npzfile.close()
    return data, labels


data, labels = load_data("./dataset/Burst_Closed World/burst_tor_200w_2500tr.npz")

num_classes = 200
train_X = np.array([])
train_Y = np.array([])

test_X = np.array([])
test_Y = np.array([])


siteName = ['tokopedia.com', 'steampowered.com', 'twitter.com', 'mediafire.com', 'tripadvisor.com', 'livejasmin.com', 'ntd.tv', 'tumblr.com', 'sharepoint.com', 'sciencedirect.com', 'w3schools.com', 'yahoo.com', 'pandora.com', 'stackoverflow.com', 'appspot.com', 'flipkart.com', 'fc2.com', 'gmx.net', 'daum.net', 'naver.com', 'goodreads.com', 'web.de', 'sourceforge.net', 'researchgate.net', 'jd.com', 'archive.org', 'doublepimp.com', 'gfycat.com', 'godaddy.com', 'tistory.com', 'zippyshare.com', 'slideshare.net', 'rt.com', 'chaturbate.com', 'theladbible.com', 'wordreference.com', 'ebay-kleinanzeigen.de', 'alibaba.com', 'instructure.com', 'softonic.com', 't.co', 'popcash.net', 'openload.co', 'amazonaws.com', 'varzesh3.com', 'weather.com', 'tribunnews.com', 'pinterest.com', 'ok.ru', 'extratorrent.cc', 'youtube.com', 'washingtonpost.com', 'liputan6.com', 'blogger.com', 'youm7.com', 'ouo.io', 'dingit.tv', 'nicovideo.jp', 'detik.com', 'webtretho.com', 'stackexchange.com', 'rambler.ru', 'amazon.com', 'yts.ag', 'wellsfargo.com', 'force.com', 'breitbart.com', 'coccoc.com', 'upornia.com', 'messenger.com', 'dailymail.co.uk', 'abs-cbn.com', 'buzzfeed.com', 'espncricinfo.com', 'exoclick.com', 'livejournal.com', 'adobe.com', 'hclips.com', 'wikimedia.org', '9gag.com', 'redtube.com', 'douyu.com', 'myway.com', 'sabah.com.tr', 'adf.ly', 'rolloid.net', 'blackboard.com', 'dictionary.com', 'whatsapp.com', 'github.com', 'ettoday.net', 'onlinesbi.com', 'wetransfer.com', 'orange.fr', 'skype.com', 'hola.com', 'microsoft.com', 'conservativetribune.com', 'apple.com', 'digikala.com', 'vk.com', 'aliexpress.com', 'ebay.com', 'mozilla.org', 'quora.com', 'bukalapak.com', 'roblox.com', 'discordapp.com', 'seasonvar.ru', 'gamepedia.com', 'yelp.com', 'wix.com', 'ask.com', 'nytimes.com', 'hatena.ne.jp', 'dailymotion.com', 'kompas.com', 'nametests.com', 'kaskus.co.id', 'spotscenered.info', 'thewhizmarketing.com', 'slack.com', 'wikipedia.org', 'goo.ne.jp', 'onclkds.com', 'evernote.com', 'speedtest.net', 'allegro.pl', 'businessinsider.com', 'twitch.tv', 'kakaku.com', 'freepik.com', 'microsoftonline.com', 'oracle.com', 'wordpress.com', 'hotmovs.com', 'vimeo.com', 'go.com', 'weebly.com', 'providr.com', 'battle.net', 'blastingnews.com', 'msn.com', 'nih.gov', 'xvideos.com', 'torrentz2.eu', 'ltn.com.tw', 'wikihow.com', 'aol.com', 'soundcloud.com', 'steamcommunity.com', 'outbrain.com', 'askcom.me', 'reimageplus.com', 'hotstar.com', 'lifebuzz.com', 'google.com', 'mail.ru', 'intuit.com', 'reddit.com', 'leboncoin.fr', 'rakuten.co.jp', 'yandex.ru', 'vice.com', 'salesforce.com', 'ikea.com', 'dropbox.com', 'uptodown.com', 'chase.com', 'bongacams.com', 'booking.com', 'deviantart.com', 'imdb.com', 'avito.ru', 'shutterstock.com', 'xhamster.com', 'doubleclick.net', 'youtube-mp3.org', 'spotify.com', 'imgur.com', 'espn.com', 'subscene.com', 'livedoor.jp', 'thesaurus.com', 'bing.com', 'popads.net', 'diply.com', 'scribd.com', 'mercadolivre.com.br', 'ameblo.jp', 'netflix.com', 'facebook.com', 'linkedin.com', 'wikia.com', 'feedly.com', 'cnn.com', 'txxx.com', 'bbc.co.uk', 'theguardian.com', 'office.com']


for i in range(0, len(siteName)):
    print("###############%d  #################当前的label是%s"%(i, list(siteName)[i]))
    data_X, data_Y = get_class_samples(data, labels, list(siteName)[i])

    data_Y[data_Y != 0] = i

    train_data_X = data_X[:1000, :]
    train_data_Y = data_Y[:1000]

    test_data_X = data_X[1000:, :]
    test_data_Y = data_Y[1000:]


    if i == 0:
        train_X = train_data_X
        train_Y = train_data_Y
    else:
        train_X = np.concatenate([train_X, train_data_X], axis=0)
        train_Y = np.concatenate([train_Y, train_data_Y], axis=0)

    if i == 0:
        test_X = test_data_X
        test_Y = test_data_Y
    else:
        test_X = np.concatenate([test_X, test_data_X], axis=0)
        test_Y = np.concatenate([test_Y, test_data_Y], axis=0)


myind_train = list(range(train_X.shape[0])) # indices
random.shuffle(myind_train)
train_X = train_X[myind_train]
train_Y = train_Y[myind_train]

train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1], 1))
train_Y = np_utils.to_categorical(train_Y, num_classes)

myind_test = list(range(test_X.shape[0])) # indices
random.shuffle(myind_test)
test_X = test_X[myind_test]
test_Y = test_Y[myind_test]

test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1], 1))
test_Y = np_utils.to_categorical(test_Y, num_classes)

np.savez('./dataset/Burst_Closed World/burst_tor_200w_2500tr_train.npz', data=train_X, labels=train_Y)
np.savez('./dataset/Burst_Closed World/burst_tor_200w_2500tr_test.npz', data=test_X, labels=test_Y)

import mockingbird_utility as mb_utility
from keras.optimizers import adam_v2, adamax_v2

num_bursts=2000
num_classes=200
learning_rate=0.002
VALIDATION_SPLIT= 0.1
NB_EPOCH = 30

input_shape=(1, num_bursts, 1)
BATCH_SIZE = 128
OPTIMIZER = adamax_v2.Adamax(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model = mb_utility.ConvNet.build(input_shape=input_shape, classes=num_classes)


model.compile(loss="categorical_crossentropy", optimizer=OPTIMIZER,metrics=["acc"])
history = model.fit(train_X, train_Y, batch_size=BATCH_SIZE, epochs=NB_EPOCH, validation_split=VALIDATION_SPLIT)

model_name = "./model/origin_AWFdata_DF_epoch=30_model.h5"
model.save(model_name)

val = model.evaluate(test_X,  test_Y)
print(val)

import matplotlib.pyplot as plt
history_dict=history.history
train_loss_values=history_dict["loss"]
val_loss_values=history_dict["val_loss"]
epochs=np.arange(1,len(train_loss_values)+1)

plt.figure(1)
plt.plot(epochs,train_loss_values,label="training loss")
plt.plot(epochs,val_loss_values,label="val loss")
plt.title(" Training and Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig("./sub_model_plot/confusion_Loss.png")

train_acc=history_dict["acc"]
val_acc=history_dict["val_acc"]

plt.figure(2)
plt.plot(epochs,train_acc,label="Training Acc")
plt.plot(epochs,val_acc,label="Val Acc")
plt.title("Training and Val Acc")
plt.xlabel("Epoch")
plt.ylabel("Acc")
plt.legend()
plt.savefig("./sub_model_plot/confusion_ACC.png")

plt.show()




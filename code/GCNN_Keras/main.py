import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
import pickle
import gzip
from os import listdir, path

from keras import backend as K
from keras.models import Sequential,Model, load_model
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape, Permute,Lambda, RepeatVector
from keras.layers.convolutional import ZeroPadding2D, AveragePooling2D, Conv2D,MaxPooling2D, Convolution1D,MaxPooling1D
from keras.layers.pooling import GlobalMaxPooling2D
from keras.layers import Merge, Input, merge
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler
from keras.layers import LSTM, SimpleRNN, GRU, TimeDistributed, Bidirectional
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import Multiply
from keras import metrics
import keras
import tensorflow as tf
import time
import os


# load data to a whole dict
# dict{key:[data, label]}
def read_dir(filedir='./eval'):
    data_dict = {}
    for subfile in listdir(filedir):
        with gzip.open(path.join(filedir, subfile), 'rb') as readsub:
            curfile = pickle.load(readsub, encoding='latin1')
            data_dict.update(curfile)
    return data_dict


def format_data(data_dict):
    n_data = len(data_dict)
    keys = data_dict.keys()
    data_arr = [data_dict[key][0] for key in keys]
    labels = [data_dict[key][1] for key in keys]
    data_arr = np.array(data_arr)
    label_arr = np.zeros((n_data, 527), dtype=np.int)
    for i in range(len(labels)):
        for idx in labels[i]:
            label_arr[i][idx] = 1
    return data_arr, label_arr


def eval(pred, label):
    mAP = average_precision_score(label, pred)
    mAUC = roc_auc_score(label, pred)
    return (mAP, mAUC)

def slice1(x):
    return x[:, :, :, 0:64]

def slice2(x):
    return x[:, :, :, 64:128]

def slice1_output_shape(input_shape):
    return tuple([input_shape[0],input_shape[1],input_shape[2],64])

def slice2_output_shape(input_shape):
    return tuple([input_shape[0],input_shape[1],input_shape[2],64])

def permute(x):
    return K.permute_dimensions(x, (0, 2, 1))

def gcnn_block(input):
    cnn = Conv2D(128, (2, 2), padding="same", activation="linear", use_bias=False)(input)
    cnn = BatchNormalization(axis=-1)(cnn)

    cnn1 = Lambda(slice1, output_shape=slice1_output_shape)(cnn)
    cnn2 = Lambda(slice2, output_shape=slice2_output_shape)(cnn)

    cnn1 = Activation('linear')(cnn1)
    cnn2 = Activation('sigmoid')(cnn2)

    out = Multiply()([cnn1, cnn2])
    return out

def outfunc(vects):
    cla, att = vects    # (N, n_time, n_out), (N, n_time, n_out)
    att = K.clip(att, 1e-7, 1.)
    out = K.sum(cla * att, axis=1) / K.sum(att, axis=1)     # (N, n_out)
    return out


def permute(x):
    return K.permute_dimensions(x, (0, 2, 1))

def permute_out_shape(input_shape):
    return tuple([input_shape[0], input_shape[2], input_shape[1]])

def my_eval(y_true, y_pred):
    return float(roc_auc_score(y_true, y_pred))

def gcnn_build(lr=0.001, opt='adam', n_class = 527, timestep=12, dim=128):
    def gcnn_build(lr=0.001, opt='adam', n_class = 527, timestep=10, dim=128):
        log = ''

        input_layer = Input(shape=(timestep, dim), name='in_layer')
        net = Reshape((timestep, dim, 1))(input_layer) #(N, 10, 128, 1)

        net = gcnn_block(net)
        net = gcnn_block(net)
        net = MaxPooling2D(pool_size=(2, 1))(net) #(N, 5, 128, 64)
        log += 'gcnn_block(3, 3), maxpool(2, 1)\n'

        net = Conv2D(256, (2, 2), padding='same', activation='relu', use_bias=True)(net)
        net = MaxPooling2D(pool_size=(5, 1))(net) #(N, 1, 128, 256)
        log += 'conv(2, 2), 256 maxpool(3, 1)\n'

        net = Reshape((128, 256))(net) #(N, ?, 128, 256)

        net = Lambda(permute, output_shape=permute_out_shape)(net) # (N, ?, 256, 128)

        rnngate1 = Bidirectional(GRU(128, activation='linear', return_sequences=True))(net)
        rnngate2 = Bidirectional(GRU(128, activation='sigmoid', return_sequences=True))(net)
        net = Multiply()([rnngate1, rnngate2])
        log += 'BiGRU 128 * 2\n'

        cla = TimeDistributed(Dense(n_class, activation='sigmoid'), name='localization_layer')(net)
        att = TimeDistributed(Dense(n_class, activation='softmax'))(net)
        out = Lambda(outfunc, output_shape=(n_class,))([cla, att])
        log += 'Attention cla att\n'

        model = Model(input_layer, out)
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'], clipvalue=0.25)
        with open('arch.txt', 'w') as f:
            model.summary(print_fn=lambda x: f.write(x))
        model.summary()
        return (model, log)

def buildlog(path='log.txt', bs=-1, epoch=-1, add='', ap=0.0, auc=0.0):
    res = ''
    log = open(path, 'a+')
    res += '---\n'
    res += time.strftime("%m-%d %H:%M:%S", time.localtime()) + '\n'
    res += str(bs) + '\n'
    res += str(epoch) + '\n'
    res += add + '\n'
    res += 'mAP: ' + str(ap) + '\n'
    res += 'mAUC: ' + str(auc) + '\n'
    log.write(res)
    log.close()
    return res

def gcnn_train(model, train_d, train_l, valid_d, valid_l, bs=64, epoch=1, logs=''):
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=5, verbose=1, mode='auto')
    model.fit(x=train_d, y=train_l, batch_size=bs, epochs=epoch, verbose=1, validation_data=(valid_d, valid_l), shuffle=True, callbacks=[early_stop])
    pred = model.predict(valid_d)
    mAP, mAUC = eval(pred, valid_l)

    print('mAP:', mAP)
    print('mAUC:', mAUC)

    print('rm ' + os.path.join(os.getcwd(), 'output.pkl'))
    os.system('rm ' + os.path.join(os.getcwd(), 'output.pkl'))
    with open('output.pkl', 'wb') as f:
        pickle.dump(pred, f)
        print('Build ' + os.path.join(os.getcwd(), 'output.pkl'))
    print(buildlog(bs=bs, epoch=epoch, add=logs, ap=mAP, auc=mAUC))
    with open('bestmodel/performance.txt', 'r+') as f:
        s = f.readline()
        if s == '':
            s = '-1.0'
        bst_ap = float(s)
        s = f.readline()
        if s == '':
            s = '-1.0'
        bst_auc = float(s)
        print('Latest mAP: ' + str(bst_ap))
        print('Latest mAUC: ' + str(bst_auc))
    print('mAP:', mAP)
    print('mAUC:', mAUC)
    if mAP >= bst_ap and mAUC >= bst_auc:
        print('Better model! Saving model at /bestmodel...')

        print('rm ' + os.path.join(os.getcwd(), 'output.pkl'))
        os.system('rm ' + os.path.join(os.getcwd(), 'bestmodel', 'output.pkl'))
        with open('bestmodel/output.pkl', 'wb') as f:
            pickle.dump(pred, f)
            print('Build ' + os.path.join(os.getcwd(), 'bestmodel', 'output.pkl'))

        model.save('bestmodel/best.model')

        buildlog(path='bestmodel/log.txt', bs=bs, epoch=epoch, add=logs, ap=mAP, auc=mAUC)

        with open('bestmodel/performance.txt', 'w+') as f:
            f.write(str(mAP) + '\n')
            f.write(str(mAUC) + '\n')
        with open('bestmodel/arch.txt', 'w+') as f:
            model.summary(print_fn=lambda x: f.write(x))
    else:
        print('Not a better model, QAQ. Saving model at /gcnn.model ...')
    model.save('gcnn.model')

def gcnn_data_transform(stdformat):
    print('Transforming data to gcnn format[0 padding]...')
    num = stdformat.shape[0]
    print(stdformat.shape)
    res = np.zeros((num, 10, 128))
    for ind in range(num):
        if ind % 5000 == 0:
            print(ind)
        d = stdformat[ind]
        s = d.copy()
        while s.shape[0] != 10:
            s = np.concatenate((s, np.zeros((1, 128))), axis=0)
        res[ind, :, :] = s
    print(res.shape)
    return res
def run(td, tl, ed, el, lr, ep, bs, gpu=-1, optimizer='adam'):
    gpu_config = '/gpu:'
    if gpu == -1:
        gpu_config = '/cpu:0'
    else:
        gpu_config += str(gpu)
    with tf.device(gpu_config):
        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        K.set_session(sess)
        model, lg = gcnn_build(lr=lr, opt=optimizer)
        gcnn_train(model, td, tl, ed, el, bs=bs, epoch=ep, logs=lg)

def calc_time(start, end):
    with open('time.txt', 'w+') as f:
        f.write(str(end-start))

if __name__ == "__main__":
    start = time.clock()
    train_data_dict = read_dir('./train')
    eval_data_dict = read_dir('./eval')

    train_data, train_l = format_data(train_data_dict)
    eval_data, eval_l = format_data(eval_data_dict)

    eval_d = gcnn_data_transform(eval_data) / 255.0
    train_d = gcnn_data_transform(train_data) / 255.0

    for method_ in ['adam']:
        for lr_ in [0.0001]:
            for bs_ in [64]:
                for ep_ in [10]:
                    tf.reset_default_graph()
                    run(train_d, train_l, eval_d, eval_l, lr=lr_, bs=bs_, ep=ep_, gpu=1, optimizer=method_)

    end = time.clock()
    calc_time(start, end)

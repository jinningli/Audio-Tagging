# This script is a toy script which gives you basic idea of loading the data provided
# Read all the bal_train data into dicts
# For Python 3+

import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
import pickle
import gzip
from os import listdir, path
import tensorflow as tf
import keras
from keras import backend as kb
from keras.layers import Input
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense
from keras.models import Model
import time
import os


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


def lstm_data_transform(stdformat):
    print('Transforming data to lstm format...')
    num = stdformat.shape[0]
    print(stdformat.shape)
    res = np.zeros((num, 10, 128))
    for ind in range(num):
        if ind % 5000 == 0:
            print(ind)
        d = stdformat[ind]
        s = d.copy()
        while s.shape[0] != 10:
            s = np.concatenate((s[0].reshape(1, 128), s), axis=0)
        res[ind, :, :] = s
    print(res.shape)
    return res


def lstm_build(times=10, dim=128, classes=527, lr=0.0001, opt='Adam'):
    log = ''
    input_layer = Input(shape=(times, dim), name='in_layer')
    log += str(times) + ' ' +  str(dim) + '\n'

    net = LSTM(512, dropout=0.8, return_sequences=True)(input_layer)
    log += 'lstm 512, 0.8\n'
    net = LSTM(256, dropout=0.8, return_sequences=True)(net)
    log += 'lstm 256, 0.8\n'
    net = LSTM(128, dropout=0.8)(net)
    log += 'lstm 128, 0.8\n'
    net = Dense(2048, activation='elu')(net)
    log += 'fc 2048 relu\n'
    net = Dense(1024, activation='elu')(net)
    log += 'fc 1024 relu\n'
    net = Dense(classes, activation='softmax')(net)
    log += 'fc ' + str(classes) + ' softmax\n'

    model = Model(input_layer, net)
    opti = keras.optimizers.Adam(lr=lr, clipvalue=0.1)
    model.compile(loss='categorical_crossentropy', optimizer=opti)
    model.summary()
    log += opt + ' ' + str(lr)

    return (model, log)


def lstm_train(data, label, validdata, validlabel, model, bs=64, epoch=1, logs=''):
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='auto')
    model.fit(data, label, epochs=epoch, validation_data=(validdata, validlabel), verbose=1, batch_size=bs, shuffle=True, callbacks=[early_stop])

    pred = model.predict(validdata)
    mAP, mAUC = eval(pred, validlabel)

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
    else:
        print('Not a better model, QAQ. Saving model at /lstm.model ...')
    model.save('lstm.model')

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

def run(td, tl, ed, el, lr, ep, bs, gpu=-1, optimizer='Adam'):
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    config.gpu_options.allow_growth = True
    tf.add_to_collection('graph_config', config)

    gpu_config = '/gpu:'
    if gpu == -1:
        gpu_config = '/cpu:0'
    else:
        gpu_config += str(gpu)
    with tf.device(gpu_config):
        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        kb.set_session(sess)
        model, lg = lstm_build(lr=lr, opt=optimizer)
        lstm_train(td, tl, ed, el, model, bs=bs, epoch=ep, logs=lg)

def calc_time(start, end):
    with open('time.txt', 'w+') as f:
        f.write(str(end-start))

if __name__ == "__main__":
    start = time.clock()
    train_data_dict = read_dir('./train')
    eval_data_dict = read_dir('./eval')

    train_data, train_label = format_data(train_data_dict)
    eval_data, eval_label = format_data(eval_data_dict)

    eval_d = lstm_data_transform(eval_data) / 255.0
    train_d = lstm_data_transform(train_data) / 255.0

    for method_ in ['Adam']:
        for lr_ in [0.0001]:
            for bs_ in [64]:
                for ep_ in [10]:
                    tf.reset_default_graph()
                    run(train_d, train_label, eval_d, eval_label, lr=lr_, bs=bs_, ep=ep_, gpu=1, optimizer=method_)

    end = time.clock()
    calc_time(start, end)

    # model, lg = lstm_build(lr=0.0005, opt='Adam')
    # model.load('bestmodel/best.model')
    # pred = model.predict(eval_d)
    # print(eval(pred, eval_label))
    # print('rm ' + os.path.join(os.getcwd(), 'output.pkl'))
    # os.system('rm ' + os.path.join(os.getcwd(), 'bestmodel', 'output.pkl'))
    # with open('bestmodel/output.pkl', 'wb') as f:
    #     pickle.dump(pred, f)
    #     print('Build ' + os.path.join(os.getcwd(), 'bestmodel', 'output.pkl'))
    # with open('bestmodel/output.pkl', 'rb') as f:
    #     pred = pickle.load(f)
    # print(eval(pred, eval_label))

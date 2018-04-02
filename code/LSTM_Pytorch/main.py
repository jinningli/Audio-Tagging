
import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
import pickle
import gzip
from os import listdir, path
import time
import os

import torch
import torch.utils.data as Data
from net import LSTMNET
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim

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

def calc_time(start, end):
    with open('time.txt', 'w+') as f:
        f.write(str(end-start))

if __name__ == "__main__":
    start = time.clock()
    train_data_dict = read_dir('./train')
    eval_data_dict = read_dir('./eval')

    train_data, train_l = format_data(train_data_dict)
    eval_data, eval_l = format_data(eval_data_dict)

    eval_d = lstm_data_transform(eval_data)/255.0
    train_d = lstm_data_transform(train_data)/255.0

    trainset = Data.TensorDataset(data_tensor=torch.from_numpy(train_d).float(), target_tensor=torch.from_numpy(train_l).float())
    trainloader = Data.DataLoader(
        dataset=trainset,  # torch TensorDataset format
        batch_size=32,  # mini batch size
        shuffle=True,
        num_workers=2,
    )
    evalset = Data.TensorDataset(data_tensor=torch.from_numpy(eval_d).float(),
                                  target_tensor=torch.from_numpy(eval_l).float())
    evalloader = Data.DataLoader(
        dataset=evalset,  # torch TensorDataset format
        batch_size=32,  # mini batch size
        shuffle=True,
        num_workers=2,
    )

    usegpu = True
    with torch.cuda.device(0):
        if usegpu:
            model = LSTMNET().cuda()
        else:
            model = LSTMNET()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # optimize all cnn parameters
        # optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
        loss_func = nn.BCELoss()
        # tag_scores = model.forward(loader)
        # print(tag_scores)

        for epoch in range(15):
            for step, (batch_x, batch_y) in enumerate(trainloader):
                if usegpu:
                    b_x = Variable(batch_x).cuda()
                    b_y = Variable(batch_y).cuda()
                else:
                    b_x = Variable(batch_x)
                    b_y = Variable(batch_y)
                output = model.forward(b_x)  # rnn output
                loss = loss_func(input=output, target=b_y)
                optimizer.zero_grad()  # clear gradients for this training step
                loss.backward()  # backpropagation, compute gradients
                torch.nn.utils.clip_grad_norm(model.parameters(), 0.4)
                optimizer.step()  # apply gradients
                if step % 100 == 0:
                    if usegpu:
                        evalpred = model.forward(Variable(torch.from_numpy(eval_d)).float().cuda())
                        evalloss = loss_func(input=evalpred, target=Variable(torch.from_numpy(eval_l)).float().cuda())
                        print (evalpred)
                    else:
                        evalpred = model.forward(Variable(torch.from_numpy(eval_d)).float())
                        evalloss = loss_func(input=evalpred, target=Variable(torch.from_numpy(eval_l)).float())
                    print ('Epoch: ' + str(epoch) + ' | ' + str(step*batch_x.size(0)) + '/' + str(len(trainloader.dataset)) + '\t' + 'Train loss: ' + str(float(loss.data.cpu().numpy())) + ' Val Loss: ' + str(float(evalloss.data.cpu().numpy())))
                    print (eval(evalpred.data.cpu().numpy(), eval_l))
        print('Saving result...')
        print('rm ' + os.path.join(os.getcwd(), 'output.pkl'))
        os.system('rm ' + os.path.join(os.getcwd(), 'output.pkl'))
        with open('output.pkl', 'wb') as f:
            pickle.dump(evalpred, f)
            print('Build ' + os.path.join(os.getcwd(), 'output.pkl'))
    end = time.clock()
    calc_time(start, end)
    # for method_ in ['Adam', 'rmsprop']:
    #     for lr_ in [0.0001, 0.0005, 0.001, 0.005, 0.00008]:
    #         for bs_ in [16, 32, 64, 128]:
    #             for ep_ in [50]:
    #                 tf.reset_default_graph()
    #                 run(train_d, train_label, eval_d, eval_label, lr=lr_, bs=bs_, ep=ep_, gpu=1, optimizer=method_)
    #

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

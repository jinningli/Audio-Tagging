import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.autograd as autograd

class LSTMNET(nn.Module):
    def __init__(self):
        super(LSTMNET, self).__init__()
        self.rnn1 = nn.LSTM(         # if use nn.RNN(), it hardly learns
            input_size=128,
            hidden_size=128,         # rnn hidden unit
            num_layers=3,           # number of rnn layer
            batch_first=True,      # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
            dropout=0.8
            )
        self.fc1 = nn.Linear(1280, 2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.fc3 = nn.Linear(1024, 527)

    def forward(self, x):
        out, _ = self.rnn1(x)
        flt = out.contiguous().view(x.size(0), -1)
        out = F.relu(self.fc1(flt))
        out = F.relu(self.fc2(out))
        out = F.sigmoid(self.fc3(out))
        return out

# class LSTMNET(nn.Module):
#
#     def __init__(self, input_dim=128, hidden_dim=512, tagset_size=527):
#         super(LSTMNET, self).__init__()
#         self.hidden_dim = hidden_dim
#         self.input_dim = input_dim
#         # The LSTM takes word embeddings as inputs, and outputs hidden states
#         # with dimensionality hidden_dim.
#         self.lstm = nn.LSTM(input_dim, hidden_dim)
#
#         # The linear layer that maps from hidden state space to tag space
#         self.hidden2tag = nn.Linear(hidden_dim * 10, tagset_size)
#         self.hidden = self.init_hidden()
#
#     def init_hidden(self):
#         # Before we've done anything, we dont have any hidden state.
#         # Refer to the Pytorch documentation to see exactly
#         # why they have this dimensionality.
#         # The axes semantics are (num_layers, minibatch_size, hidden_dim)
#         return (autograd.Variable(torch.zeros(1, 1, self.hidden_dim)),
#                 autograd.Variable(torch.zeros(1, 1, self.hidden_dim)))
#
#     def forward(self, sentence):
#         lstm_out, self.hidden = self.lstm(
#             sentence.view(-1, 10, self.input_dim), self.hidden)
#
#         tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
#         tag_scores = F.sigmoid(tag_space)
#         return tag_scores

# class LSTMNET(nn.Module):
#     def __init__(self):
#         super(LSTMNET, self).__init__()
#
#         self.rnn = nn.LSTM(         # if use nn.RNN(), it hardly learns
#             input_size=128,
#             hidden_size=64,         # rnn hidden unit
#             num_layers=1,           # number of rnn layer
#             batch_first=True,       # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
#         )
#
#         self.out = nn.Linear(64, 10)
#
#     def forward(self, x):
#         # x shape (batch, time_step, input_size)
#         # r_out shape (batch, time_step, output_size)
#         # h_n shape (n_layers, batch, hidden_size)
#         # h_c shape (n_layers, batch, hidden_size)
#         r_out, (h_n, h_c) = self.rnn(x, None)   # None represents zero initial hidden state
#
#         # choose r_out at the last time step
#         out = self.out(r_out[:, -1, :])
#         return out

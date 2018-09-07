from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import norm_col_init, weights_init, Merger
import vaderSentiment
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pickle
import numpy as np

class diagLinear(nn.Module):
    def __init__(self, in_features):
        super(diagLinear, self).__init__()
        self.in_features = in_features
        self.weight = nn.Parameter(torch.Tensor(in_features))
        self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        self.weight = torch.nn.Parameter(torch.zeros(self.in_features))

    def forward(self, input):
        return F.linear(input = input, weight = torch.diag(self.weight), bias=None)


from torch.nn.utils.rnn import pack_padded_sequence

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, max_seq_length=20):
        """Set the hyper-parameters and build the layers."""
        super(DecoderRNN, self).__init__()
            # Load vocabulary wrapper
        with open('wordDict.pkl', 'rb') as f:
            self.idx2word = pickle.load(f)
        self.analyser = SentimentIntensityAnalyzer()

        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.max_seg_length = max_seq_length
    
    def forward_old(self, features, captions, lengths):
        """Decode image feature vectors and generates captions."""
        embeddings = self.embed(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True) 
        hiddens, _ = self.lstm(packed)
        outputs = self.linear(hiddens[0])
        return outputs
    
    def forward(self, features):
        """Generate captions for given image features using greedy search."""
        sampled_ids = []
        values = []
        inputs = features.unsqueeze(1)
        states = None
        for i in range(self.max_seg_length):
            hiddens, states = self.lstm(inputs, states)          # hiddens: (batch_size, 1, hidden_size)
            outputs = self.linear(hiddens.squeeze(1))            # outputs:  (batch_size, vocab_size)
            value, predicted = outputs.max(1)                        # predicted: (batch_size)
            values.append(value)
            sampled_ids.append(predicted)
            inputs = self.embed(predicted)                       # inputs: (batch_size, embed_size)
            inputs = inputs.unsqueeze(1)                         # inputs: (batch_size, 1, embed_size)
        sampled_ids = torch.stack(sampled_ids, 1)                # sampled_ids: (batch_size, max_seq_length)
        values = torch.stack(values, 1)                # sampled_ids: (batch_size, max_seq_length)
        return sampled_ids, values


class A3Clstm(torch.nn.Module):
    def __init__(self, num_inputs, action_space):
        super(A3Clstm, self).__init__()
        self.conv1 = nn.Conv2d(num_inputs, 32, 5, stride=1, padding=2)
        self.maxp1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 32, 5, stride=1, padding=1)
        self.maxp2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(32, 64, 4, stride=1, padding=1)
        self.maxp3 = nn.MaxPool2d(2, 2)
        self.conv4 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.maxp4 = nn.MaxPool2d(2, 2)
        
        self.lstm = nn.LSTMCell(1024, 512)
        num_outputs = action_space.n
        self.critic_linear = nn.Linear(512, 1)
        self.actor_linear = nn.Linear(512, num_outputs)

        self.apply(weights_init)
        relu_gain = nn.init.calculate_gain('relu')
        self.conv1.weight.data.mul_(relu_gain)
        self.conv2.weight.data.mul_(relu_gain)
        self.conv3.weight.data.mul_(relu_gain)
        self.conv4.weight.data.mul_(relu_gain)
        self.actor_linear.weight.data = norm_col_init(
            self.actor_linear.weight.data, 0.01)
        self.actor_linear.bias.data.fill_(0)
        self.critic_linear.weight.data = norm_col_init(
            self.critic_linear.weight.data, 1.0)
        self.critic_linear.bias.data.fill_(0)

        self.lstm.bias_ih.data.fill_(0)
        self.lstm.bias_hh.data.fill_(0)
        self.train()
        
    def add_app(self, num_inputs):
        self.app1 = nn.Conv2d(num_inputs, 4, 5, stride=1, padding=1)
        self.app2 = nn.Linear(24336, 1024)
        self.app3 = Merger(1024)
        self.forward = self.forward_app
    
    def add_lstm_app(self, num_inputs):
        self.app1 = nn.Linear(1024, 256)
        self.app2 = DecoderRNN(256, 512, 9956, 1, max_seq_length=40)
        self.app2.load_state_dict(torch.load('decoder-5-3000.pkl'))
        for param in self.app2.parameters():
            param.requires_grad = False
        
        self.app3 = nn.Linear(40, 512)
        self.app4 = Merger(512)
        self.forward = self.forward_lstm_app
    
    def remove_app(self):
        self.forward = self.old_forward
        for param in self.parameters():
            param.requires_grad = False
        
            
    
    def sentence_and_sentiment(self, inputs):
        inputs, (hx, cx) = inputs
        app_x = F.relu(self.maxp1(self.conv1(inputs)))
        app_x = F.relu(self.maxp2(self.conv2(app_x)))
        app_x = F.relu(self.maxp3(self.conv3(app_x)))
        app_x = F.relu(self.maxp4(self.conv4(app_x)))
        app_x = app_x.view(app_x.size(0), -1)
        app_x = F.relu(self.app1(app_x))
        word_indices, app_x = self.app2(app_x)
        sampled_caption = []
        for word_id in word_indices.detach().cpu().numpy().reshape(-1):  
            word = self.app2.idx2word[word_id]
            if word == '<end>':
                break
            sampled_caption.append(word)
        sentence = ' '.join(sampled_caption)
        snt = self.app2.analyser.polarity_scores(sentence)
        return sentence, snt
    
    def forward_lstm_app(self, inputs):
        inputs, (hx, cx) = inputs
        
        x = F.relu(self.maxp1(self.conv1(inputs)))
        x = F.relu(self.maxp2(self.conv2(x)))
        x = F.relu(self.maxp3(self.conv3(x)))
        x = F.relu(self.maxp4(self.conv4(x)))
        
        x = x.view(x.size(0), -1)
        app_x = x.clone()
        
        # print('(app_x / x) = {}'.format((app_x / (x + 0.0000001)).abs().mean()))
        hx, cx = self.lstm(x, (hx, cx))
        
        x = hx
        
        app_x = app_x.view(app_x.size(0), -1)
        app_x = F.relu(self.app1(app_x))
        word_indices, app_x = self.app2(app_x)
        app_x = F.relu(self.app3(app_x).float())
        x = self.app4(app_x, x)
        
        return self.critic_linear(x), self.actor_linear(x), (hx, cx)
            
    
    def forward_app(self, inputs):
        inputs, (hx, cx) = inputs
        app_x = inputs.clone()        
        
        x = F.relu(self.maxp1(self.conv1(inputs)))
        x = F.relu(self.maxp2(self.conv2(x)))
        x = F.relu(self.maxp3(self.conv3(x)))
        x = F.relu(self.maxp4(self.conv4(x)))

        x = x.view(x.size(0), -1)

        app_x = F.relu(self.app1(app_x))
        app_x = app_x.view(x.size(0), -1)
        app_x = F.relu(self.app2(app_x)) 
        x = self.app3(app_x, x)
        # print('(app_x / x) = {}'.format((app_x / (x + 0.0000001)).abs().mean()))

        hx, cx = self.lstm(x, (hx, cx))

        x = hx

        return self.critic_linear(x), self.actor_linear(x), (hx, cx)
        
    

    def forward(self, inputs):
        inputs, (hx, cx) = inputs
        x = F.relu(self.maxp1(self.conv1(inputs)))
        x = F.relu(self.maxp2(self.conv2(x)))
        x = F.relu(self.maxp3(self.conv3(x)))
        x = F.relu(self.maxp4(self.conv4(x)))
        x = x.view(x.size(0), -1)
        hx, cx = self.lstm(x, (hx, cx))
        x = hx
        return self.critic_linear(x), self.actor_linear(x), (hx, cx)

    def old_forward(self, inputs):
        inputs, (hx, cx) = inputs
        x = F.relu(self.maxp1(self.conv1(inputs)))
        x = F.relu(self.maxp2(self.conv2(x)))
        x = F.relu(self.maxp3(self.conv3(x)))
        x = F.relu(self.maxp4(self.conv4(x)))
        x = x.view(x.size(0), -1)
        hx, cx = self.lstm(x, (hx, cx))
        x = hx
        return self.critic_linear(x), self.actor_linear(x), (hx, cx)

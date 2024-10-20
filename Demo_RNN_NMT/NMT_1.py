
from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import re
import random

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

from File_Process import *  #导入数据处理相关函数

use_cuda = torch.cuda.is_available()    #判断cuda可用的符号变量

## 序列到序列（seq2seq）编码 
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        # GRU是一个RNN的一种基本网络单元，捕捉长期依赖关系，减轻传统RNN的梯度消失
        self.embedding = nn.Embedding(input_size, hidden_size)  #词嵌入用于反应词间关系及标识词语特征，将每个单词编码为一个 hidden_size 维的向量
        self.gru = nn.GRUCell(hidden_size, hidden_size) #这里的hidden_size即是embedding之后的每个词的向量维数=中间隐藏节点个数，256维

    def forward(self, input, hidden): #一次输入一个单词，并输出结果及隐藏向量，与传统的前向传播相同，没有采用一个序列进行输入的写法。       
        output = self.embedding(input)
        hidden = self.gru(output, hidden)
        return hidden

    def initHidden(self):
        result = torch.zeros(1, self.hidden_size) #由于隐藏向量也是（layer*direction，hidden_size）格式，因此要写成（1，256）
        if use_cuda:
            return result.cuda()
        else:
            return result

## 解码器-输出
class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRUCell(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):
        output = self.embedding(input) #（1，256）的数据作为输入
        output = F.relu(output)
        hidden = self.gru(output, hidden)
        output = F.log_softmax(self.out(hidden))
        return output, hidden

    def initHidden(self):
        result = torch.zeros(1, self.hidden_size)
        if use_cuda:
            return result.cuda()
        else:
            return result

####=========convert pairs to indexs========================================================
## lang：包含词汇表及其索引 LangEmbed类
## 将句子转换为索引列表、将索引列表转换为张量，并将成对的句子（输入和输出）转换为索引张量对
def sentence2index(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]

## 将句子转换为索引列表，并添加结束符（EOS），然后转换为 PyTorch 的 LongTensor
def indexesFromSentence(lang, sentence):
    indexes = sentence2index(lang, sentence)
    indexes.append(EOS_token)
    result = torch.LongTensor(indexes).view(-1, 1)
    if use_cuda:
        return result.cuda()
    else:
        return result

## 将成对的句子分别转换为输入张量和目标张量
def indexesFromPair(pair):
    inputs= indexesFromSentence(input_lang, pair[1])  
    targets = indexesFromSentence(output_lang, pair[0])
    return (inputs, targets)
####=================================================================

teacher_forcing_ratio = 0.5

## 单次训练代码  将一个输入序列传递给编码器，然后将编码器的最后一个隐藏状态传递给解码器，
## 通过解码器生成输出序列，并计算输出序列与目标序列之间的损失，最后通过反向传播更新模型参数
def train(inputs, targets, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = inputs.size()[0] 
    target_length = targets.size()[0]

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size) #前面定义了max_length=10,即不超过十个单词
    encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs # 如果不是基于attention机制，这个变量将不需要

    loss = 0
    
    for ei in range(input_length):
        encoder_hidden = encoder(inputs[ei], encoder_hidden)
    
    decoder_input = torch.LongTensor([SOS_token])
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input
    decoder_hidden = encoder_hidden

    # 判断是否使用教师强制  (指使用真实的上一步输出做下一步输入,收敛更快)
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
    if use_teacher_forcing:        
        for di in range(target_length):            
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            loss += criterion(decoder_output, targets[di])
            decoder_input = targets[di]  # 输入为上一步真实目标

    else:
        # 没有 Teacher Forcing  (把上一次预测作为输入)
        for di in range(target_length):
            
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            topv, topi = decoder_output.data.topk(1) #返回输出中概率最高的词
            ni = topi[0][0]

            decoder_input = torch.LongTensor([ni])
            decoder_input = decoder_input.cuda() if use_cuda else decoder_input

            loss += criterion(decoder_output, targets[di])
            if ni == EOS_token:
                break

    loss.backward()
    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length #算出平均每一个词的错误是多少

##循环训练迭代  ps：n_iters：训练迭代次数。
def trainIters(encoder, decoder, n_iters, print_every=500, learning_rate=0.01):

    print_loss_total = 0

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    
    criterion = nn.NLLLoss()    #负对数似然损失（NLLLoss）

    for iter in range(1, n_iters + 1):
        random.shuffle(pairs)
        training_pairs = [indexesFromPair(pair) for pair in pairs]
        
        for idx,training_pair in enumerate(training_pairs):
            input_index = training_pair[0]
            target_index = training_pair[1]

            loss = train(input_index, target_index, encoder,
                         decoder, encoder_optimizer, decoder_optimizer, criterion)
        
            print_loss_total += loss
    
            if idx % print_every == 0:

                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                print('iteration:%s, idx:%d, average loss:%.4f' % (iter,idx,print_loss_avg))

## 定义隐藏向量256维
hidden_size = 256
encoder = EncoderRNN(input_lang.n_words, hidden_size)
decoder = DecoderRNN(hidden_size,output_lang.n_words)

if use_cuda:
    encoder = encoder.cuda()
    decoder = decoder.cuda()

## 迭代10次
trainIters(encoder, decoder, 10)
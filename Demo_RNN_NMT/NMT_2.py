#coding=utf-8

from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import re
import random

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

SOS_token = 0
EOS_token = 1

class LangEmbed:
    def __init__(self,name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

MAX_LENGTH = 10
eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)

def filterPairs(pairs):
    p = []
    for pair in pairs:
        if len(pair[0].split(' ')) < MAX_LENGTH and len(pair[1].split(' ')) < MAX_LENGTH and pair[0].startswith(eng_prefixes):
            p.append(pair)
    return p

def prepareData():
    lines = open('data/eng-fra.txt', encoding='utf-8').read().strip().split('\n')
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]

    fra_lang = LangEmbed("fra")
    eng_lang = LangEmbed("eng")

    print("Read %s sentence pairs" % len(pairs))
    
    pairs = filterPairs(pairs)
    print("Trimmed to %s sentence pairs" % len(pairs))
    
    for pair in pairs:
        fra_lang.addSentence(pair[1])
        eng_lang.addSentence(pair[0])
    print("Number of Words:")
    print("eng:", eng_lang.n_words)
    print("fra:", fra_lang.n_words)
    return fra_lang, eng_lang, pairs


input_lang, output_lang, pairs = prepareData()

print("English LangEmbed:")
for i in range(10):    
    print(i,":",output_lang.index2word[i])
print("...")

print("French LangEmbed:")
for i in range(10):    
    print(i,":",input_lang.index2word[i])
print("...")

print(random.choice(pairs))

use_cuda = torch.cuda.is_available()

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        
        output = self.embedding(input).view(1, 1, -1)
        output,hidden = self.gru(output, hidden)
        return output,hidden

    def initHidden(self):
        result = torch.zeros(1,1,self.hidden_size)
        if use_cuda:
            return result.cuda()
        else:
            return result
        
## 采用注意力机制，编码过程是将其处理成隐藏向量和输入，无需更改；该机制主要应用于解码时，
## 通过一个离散概率分布进行嵌入向量和处理过的隐藏向量的加权，生成新的语义向量
class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size      #GRU的隐藏层大小
        self.output_size = output_size      #输出词汇表的大小
        self.dropout_p = dropout_p          #dropout层的丢弃概率，用于防止过拟合
        self.max_length = max_length        #输入序列的最大长度

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)   #词嵌入层，将单词索引映射到向量
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)        #计算注意力权重的线性层（利用拼接向量计算）
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)   #结合嵌入向量和注意力应用结果的线性层
        self.dropout = nn.Dropout(self.dropout_p)                           #dropout层，用于正则化
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)               #处理序列数据
        self.out = nn.Linear(self.hidden_size, self.output_size)            #将GRU的输出映射到目标词汇表

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1) #转换为嵌入向量并应用dropout
        embedded = self.dropout(embedded)

        # 输入的嵌入向量和上一个时间步的隐藏状态拼接，计算注意力权重
        attn_weights = F.softmax(self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        # 注意力权重应用于编码器输出，计算上下文向量（bmm是矩阵乘法函数，进行线性矩阵运算也就是应用）
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))
        
        # 当前输入的嵌入向量和上下文向量（即应用注意力权重之后的结果）拼接，通过线性层和ReLU激活函数处理
        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)
        output = F.relu(output)
        
        # 处理后的输出传递给GRU，得到新的输出和隐藏状态
        output, hidden = self.gru(output, hidden)

        # 通过线性层映射到目标词汇表，使用log softmax计算类别概率
        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    # 初始化隐藏层
    def initHidden(self):
        result = torch.zeros(1, 1, self.hidden_size)
        if use_cuda:
            return result.cuda()
        else:
            return result

####=========convert pairs to indexs========================================================
def sentence2index(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]


def indexesFromSentence(lang, sentence):
    indexes = sentence2index(lang, sentence)
    indexes.append(EOS_token)
    result = torch.LongTensor(indexes).view(-1, 1)
    if use_cuda:
        return result.cuda()
    else:
        return result

def indexesFromPair(pair):
    inputs= indexesFromSentence(input_lang, pair[1])
    targets = indexesFromSentence(output_lang, pair[0])
    return (inputs, targets)
####=================================================================

## 训练和之前一样不变
def trainIters(encoder, decoder, n_iters, print_every=1000, learning_rate=0.01):

    print_loss_total = 0

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    
    criterion = nn.NLLLoss()

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


teacher_forcing_ratio = 0.5

def train(inputs, targets, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = inputs.size()[0] 
    target_length = targets.size()[0]

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size) 
    encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs 

    loss = 0
    
    for ei in range(input_length):

        encoder_output, encoder_hidden = encoder(inputs[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0][0]
    
    decoder_input = torch.LongTensor([SOS_token])

    decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        
        for di in range(target_length):

            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)

            loss += criterion(decoder_output, targets[di])
            decoder_input = targets[di]  # Teacher forcing

    else:
        for di in range(target_length):
            
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            
            topv, topi = decoder_output.data.topk(1) #返回最大的（数据，序号）
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


hidden_size = 256
encoder = EncoderRNN(input_lang.n_words, hidden_size)
decoder = AttnDecoderRNN(hidden_size,output_lang.n_words,dropout_p=0.1)

if use_cuda:
    encoder = encoder.cuda()
    decoder = decoder.cuda()

trainIters(encoder, decoder, 10, print_every=500)


## 评估函数
def evaluate(encoder, decoder, sentence, max_length=MAX_LENGTH):
    inputs = indexesFromSentence(input_lang, sentence)  #句子转索引
    input_length = inputs.size()[0]
    encoder_hidden = encoder.initHidden()

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size)
    encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs

    # 编码器前向传播
    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(inputs[ei],encoder_hidden)
        encoder_outputs[ei] = encoder_outputs[ei] + encoder_output[0][0]    #当前每一步输出记录

    decoder_input = torch.LongTensor([[SOS_token]]) # 初始化解码器的输入为起始符号
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    decoder_hidden = encoder_hidden # 隐藏层相同

    decoded_words = []
    decoder_attentions = torch.zeros(max_length, max_length)

    # 每个单词解码处理 解码器前向传播
    for di in range(max_length):
        decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
        decoder_attentions[di] = decoder_attention.data
        topv, topi = decoder_output.data.topk(1)    # 获取输出中概率最高的单词及其索引
        ni = topi[0][0].item()      # 获取概率最高的单词的索引
        if ni == EOS_token:         # 预测的单词是结束符号（EOS_token），则停止解码
            decoded_words.append('<EOS>')
            break
        else:
            decoded_words.append(output_lang.index2word[ni])    # 预测的单词作为下一个时间步的输入

        decoder_input = torch.LongTensor([[ni]])
        decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    return decoded_words, decoder_attentions[:di + 1]

## 随机测试评估函数 n为随机选择句子的个数
def evaluateRandomly(encoder, decoder, n=10):
    for i in range(n):
        pair = random.choice(pairs)
        print('Fra_Tar: ', pair[1])
        print('Eng_Source: ', pair[0])
        output_words, attentions = evaluate(encoder, decoder, pair[1])
        output_sentence = ' '.join(output_words)        # 输出单词序列连接成句子
        print('Pre: ', output_sentence)
        print('\n')

evaluateRandomly(encoder, decoder)
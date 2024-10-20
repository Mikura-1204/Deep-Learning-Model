from __future__ import unicode_literals, print_function, division    #兼容版本
from io import open
import unicodedata
import re
import random

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

SOS_token = 0       #0/1两序号特殊标识文件开头和结尾
EOS_token = 1

## NLP词嵌入、词典构建 常见的词与唯一序号依次对应
class LangEmbed:
    def __init__(self,name):
        self.name = name
        self.word2index = {}                        #词对序号的字典
        self.word2count = {}                        #词对出现次数的字典
        self.index2word = {0: "SOS", 1: "EOS"}      #序号对词的字典
        self.n_words = 2                            #长度

    def addSentence(self, sentence):                #按空格拆分句子中的词
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):                        #新词导入
        if word not in self.word2index:
            self.word2index[word] = self.n_words    #词对序号
            self.word2count[word] = 1               #出现次数置1
            self.index2word[self.n_words] = word    #序号对词
            self.n_words += 1                       #序号+1
        else:
            self.word2count[word] += 1              #重复则次数+1

## 转ASCII格式 去标点 转小写
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

#====================================================
## 限制句子长度和格式，必须以下述开头  过滤
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
#=========================================================

## NLP中读取文件句子对（常用于语言转换）,并转换存入词汇表
def prepareData():
    #打开文件并按行分割
    lines = open('./eng-fra.txt', encoding='utf-8').read().strip().split('\n')   

    # 每行按制表符分成对应的两句并标准化
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]

    #英语法语分别存放
    fra_lang = LangEmbed("fra")
    eng_lang = LangEmbed("eng")

    print("Read %s sentence pairs" % len(pairs))
    
    #过滤不符句子
    pairs = filterPairs(pairs)
    print("Trimmed to %s sentence pairs" % len(pairs))
    
    #存入一对，英语在前法语在后
    for pair in pairs:
        fra_lang.addSentence(pair[1])
        eng_lang.addSentence(pair[0])
    print("Number of Words:")
    print("eng:", eng_lang.n_words)
    print("fra:", fra_lang.n_words)
    return fra_lang, eng_lang, pairs

## 打印前10句存入的句子 
print("English LangEmbed:")
for i in range(10):    
    print(i,":",output_lang.index2word[i])
print("...")

print("French LangEmbed:")
for i in range(10):    
    print(i,":",input_lang.index2word[i])
print("...")
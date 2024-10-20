from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True          #服务器载入data附加代码

import torch
import torch.nn as nn
import torch.optim as optim                     #算法优化库
import torchvision
import torchvision.transforms as transforms     #图像处理库

import json
import os
from torchvision import datasets
import matplotlib.pyplot as plt

## 定义VGG基本形式块，根据结构是一次Conv后一次Relu作为一次特征提取，块特征提取结束后进行最大池化
def VGG_Block(Conv_Num, in_channel, out_channel):
    layers = []
    
    #做各层指定次数的特征提取；nn卷积无需输入权重，且保证上级out大小等于下级in
    for _ in range(Conv_Num):                                                          
        layers.append(nn.Conv2d(in_channel, out_channel, kernel_size = 3, padding = 1))
        layers.append(nn.ReLU(inplace=True))
        in_channel = out_channel
        
    #卷积提取特征结束后进行最大池化，压缩特征
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    
    #最后Sequential作为容器，打包以上操作为一个整块
    return nn.Sequential(*layers)

##构建VGG16网络类，常用nn.Module类，包含了给定的初始化方式和前向传播，可修改
class VGG_16(nn.Module):
    
    #初始化方法，用于定义网络结构和块参数
    def __init__(self):                  
        super(VGG_16, self).__init__()
        
        #五层卷积层对应参数，逐层通道数*2直至512
        self.Conv1 = VGG_Block(2, 3, 64)      
        self.Conv2 = VGG_Block(2, 64, 128)
        self.Conv3 = VGG_Block(3, 128, 256)
        self.Conv4 = VGG_Block(3, 256, 512)
        self.Conv5 = VGG_Block(3, 512, 512)
        
        #前两层固定全连接层参数（前为输入特征，后为输出特征），最后一层输出特征为分类数【此处5类花的分类】
        self.Fc1 = nn.Linear(512 * 7 * 7, 4096)
        self.Fc2 = nn.Linear(4096, 4096)
        self.Fc3 = nn.Linear(4096, 5)
        
        #正则化防止过拟合，适当丢弃参数，取0-1
        self.Dropout = nn.Dropout(0.2)
        
    #输入前向传播方法，即输入在这个网络中的传播顺序
    def forward(self, x):
        
        #依次过五个卷积层提取特征
        x = self.Conv1(x)
        x = self.Conv2(x)
        x = self.Conv3(x)
        x = self.Conv4(x)
        x = self.Conv5(x)
        
        #展平成一维进全连接
        x = x.view(x.size(0), -1)
        x = self.Dropout(self.Fc1(x))
        x = self.Dropout(self.Fc2(x))
        x = self.Fc3(x)
        return x
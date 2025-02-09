import torch
import torch.nn as nn
import torch.nn.functional as F
 
# 定义GoogLeNet网络模型
class GoogLeNet(nn.Module):
    # init()：进行初始化，申明模型中各层的定义
    # num_classes：需要分类的类别个数 [Change this !]
    # aux_logits：训练过程是否使用辅助分类器，init_weights：是否对网络进行权重初始化
    # ciallo: change to init True to see its effection
    def __init__(self, num_classes=5, aux_logits=True, init_weights=True):
        super(GoogLeNet, self).__init__()
        self.aux_logits = aux_logits
 
        self.conv1 = BasicConv2d(3, 64, kernel_size=7, stride=2, padding=3)
        # ceil_mode=true时，将不够池化的数据自动补足NAN至kernel_size大小
        self.maxpool1 = nn.MaxPool2d(3, stride=2, ceil_mode=True)
 
        self.conv2 = BasicConv2d(64, 64, kernel_size=1)
        self.conv3 = BasicConv2d(64, 192, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        # ciallo: args as the Inception1
        self.inception3a = Inception(192, 64, 96, 128, 16, 32, 32)
        # ciallo: pooling dont change channel 
        #         this input channel 256 is the sum of forward channel
        self.inception3b = Inception(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(3, stride=2, ceil_mode=True)
 
        self.inception4a = Inception(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = Inception(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = Inception(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = Inception(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = Inception(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(3, stride=2, ceil_mode=True)
 
        self.inception5a = Inception(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = Inception(832, 384, 192, 384, 48, 128, 128)
 
        # 如果为真，则使用分类器
        if self.aux_logits:
            self.aux1 = InceptionAux(512, num_classes)
            self.aux2 = InceptionAux(528, num_classes)
        # AdaptiveAvgPool2d：自适应平均池化，指定输出（H，W）
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(1024, num_classes)
        # 如果为真，则对网络参数进行初始化
        if init_weights:
            self._initialize_weights()
 
    # forward()：定义前向传播过程,描述了各层之间的连接关系
    def forward(self, x):
        # N x 3 x 224 x 224
        x = self.conv1(x)
        # N x 64 x 112 x 112
        x = self.maxpool1(x)
        # N x 64 x 56 x 56
        x = self.conv2(x)
        # N x 64 x 56 x 56
        x = self.conv3(x)
        # N x 192 x 56 x 56
        x = self.maxpool2(x)
 
        # N x 192 x 28 x 28
        x = self.inception3a(x)
        # N x 256 x 28 x 28
        x = self.inception3b(x)
        # N x 480 x 28 x 28
        x = self.maxpool3(x)
        # N x 480 x 14 x 14
        x = self.inception4a(x)
        # N x 512 x 14 x 14
        # 设置.train()时为训练模式，self.training=True
        if self.training and self.aux_logits:
            aux1 = self.aux1(x)
 
        x = self.inception4b(x)
        # N x 512 x 14 x 14
        x = self.inception4c(x)
        # N x 512 x 14 x 14
        x = self.inception4d(x)
        # N x 528 x 14 x 14
        if self.training and self.aux_logits:
            aux2 = self.aux2(x)
 
        x = self.inception4e(x)
        # N x 832 x 14 x 14
        x = self.maxpool4(x)
        # N x 832 x 7 x 7
        x = self.inception5a(x)
        # N x 832 x 7 x 7
        x = self.inception5b(x)
        # N x 1024 x 7 x 7
 
        x = self.avgpool(x)
        # N x 1024 x 1 x 1
        x = torch.flatten(x, 1)
        # N x 1024
        x = self.dropout(x)
        x = self.fc(x)
        # N x 1000 (num_classes)
        # ciallo: two return means that when mode is train, it return two output to strenthen train effection
        #         but when predict it just has one out
        if self.training and self.aux_logits:
            return x, aux2, aux1
        return x
 
    # 网络结构参数初始化
    def _initialize_weights(self):
        # 遍历网络中的每一层
        for m in self.modules():
            # isinstance(object, type)，如果指定的对象拥有指定的类型，则isinstance()函数返回True
            # 如果是卷积层
            if isinstance(m, nn.Conv2d):
                # Kaiming正态分布方式的权重初始化
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                # 如果偏置不是0，将偏置置成0，对偏置进行初始化
                if m.bias is not None:
                    # torch.nn.init.constant_(tensor, val)，初始化整个矩阵为常数val
                    nn.init.constant_(m.bias, 0)
            # 如果是全连接层
            elif isinstance(m, nn.Linear):
                # init.normal_(tensor, mean=0.0, std=1.0)，使用从正态分布中提取的值填充输入张量
                # 参数：tensor：一个n维Tensor，mean：正态分布的平均值，std：正态分布的标准差
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
 
# 基础卷积层（卷积 + ReLU）
class BasicConv2d(nn.Module):
    # init()：进行初始化，申明模型中各层的定义
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        
        # ciallo: **kwargs 是 Python 中的一种语法，用于将任意数量的关键字参数传递给一个函数。
        # 在这个例子中，它将所有传递给 **kwargs 的关键字参数转发给 nn.Conv2d 的构造函数
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        # ReLU(inplace=True)：将tensor直接修改，不找变量做中间的传递，节省运算内存，不用多存储额外的变量
        self.relu = nn.ReLU(inplace=True)
 
    # 前向传播过程
    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x
 
# Inception结构 是标准的第一个块结构
# ciallo: arg explain below:
#         in_channels：输入特征图的通道数。
#         ch1x1：1x1卷积层的输出通道数。
#         ch3x3red：3x3卷积层前的1x1卷积层的输出通道数（降维）。
#         ch3x3：3x3卷积层的输出通道数。
#         ch5x5red：5x5卷积层前的1x1卷积层的输出通道数（降维）。
#         ch5x5：5x5卷积层的输出通道数。
#         pool_proj：最大池化层后的1x1卷积层的输出通道数。
class Inception(nn.Module):
    # init()：进行初始化
    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj):
        super(Inception, self).__init__()
        # 分支1，单1x1卷积层
        self.branch1 = BasicConv2d(in_channels, ch1x1, kernel_size=1)
        # 分支2，1x1卷积层后接3x3卷积层
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, ch3x3red, kernel_size=1),
            # 保证输出大小等于输入大小
            BasicConv2d(ch3x3red, ch3x3, kernel_size=3, padding=1)
        )
        # 分支3，1x1卷积层后接5x5卷积层
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channels, ch5x5red, kernel_size=1),
            # 保证输出大小等于输入大小
            BasicConv2d(ch5x5red, ch5x5, kernel_size=5, padding=2)
        )
        # 分支4，3x3最大池化层后接1x1卷积层
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            BasicConv2d(in_channels, pool_proj, kernel_size=1)
        )
    # forward()：定义前向传播过程,描述了各层之间的连接关系
    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
 
        # 在通道维上连结输出
        outputs = [branch1, branch2, branch3, branch4]
        # cat()：在给定维度上对输入的张量序列进行连接操作,所以通道数是几个加起来的总和
        return torch.cat(outputs, 1)
 
# 辅助分类器 分别在inception(4a)和inception(4d)
class InceptionAux(nn.Module):
    # init()：进行初始化
    def __init__(self, in_channels, num_classes):
        super(InceptionAux, self).__init__()
        self.averagePool = nn.AvgPool2d(kernel_size=5, stride=3)
        self.conv = BasicConv2d(in_channels, 128, kernel_size=1)
        # 上一层output[batch, 128, 4, 4]，128X4X4=2048
        self.fc1 = nn.Linear(2048, 1024)
        
        # ciallo: out Probability of every class
        self.fc2 = nn.Linear(1024, num_classes)
 
    # 前向传播过程
    def forward(self, x):
        # 输入：分类器1：Nx512x14x14，分类器2：Nx528x14x14
        x = self.averagePool(x)
        # 输入：分类器1：Nx512x14x14，分类器2：Nx528x14x14
        x = self.conv(x)
        # 输入：N x 128 x 4 x 4
        x = torch.flatten(x, 1)
        # 设置.train()时为训练模式，self.training=True
        x = F.dropout(x, 0.5, training=self.training)
        # 输入：N x 2048
        x = F.relu(self.fc1(x), inplace=True)
        x = F.dropout(x, 0.5, training=self.training)
        # 输入：N x 1024
        x = self.fc2(x)
        # 返回值：N*num_classes
        return x
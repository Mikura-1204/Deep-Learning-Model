import torch.nn as nn
#这个.nn库里面有广泛的损失和激活、池化等辅助函数  Ciallo～(∠・ω< )⌒☆
import torch


# ResNet18/34的残差结构，用的是2个3x3的卷积   实现了ResNet中的基本残差块
class BasicBlock(nn.Module):
    expansion = 1  # 残差结构中，主分支的卷积核个数是否发生变化，不变则为1

    def __init__(self, in_channel, out_channel, stride=1, downsample=None):  # downsample对应虚线残差连接，当两块之间输入输出通道不等时，给一个下采样使得两者通道数相等
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample
    
    def forward(self, x):
        identity = x
        if self.downsample is not None:  # 虚线残差结构，需要下采样
            identity = self.downsample(x)  # 捷径分支 short cut

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity        ##捷径分支的相加 输出加原输入
        out = self.relu(out)

        return out

# ResNet50/101/152的残差结构，用的是1x1+3x3+1x1的卷积   实现了ResNet中的瓶颈残差块
class Bottleneck(nn.Module):
    expansion = 4  # 残差结构中第三层卷积核个数是第一/二层卷积核个数的4倍

    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=1, stride=1, bias=False)  # 1*1卷积 通道数从in_channel压缩到out_channel
        self.bn1 = nn.BatchNorm2d(out_channel)
        # -----------------------------------------
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channel)
        # -----------------------------------------
        self.conv3 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel * self.expansion,
                               kernel_size=1, stride=1, bias=False)  # unsqueeze channels
        self.bn3 = nn.BatchNorm2d(out_channel * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)  # 捷径分支 short cut

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    # block = BasicBlock or Bottleneck
    # block_num为残差结构中conv2_x~conv5_x中残差块个数，是一个列表
    def __init__(self, block, blocks_num, num_classes=1000, include_top=True):
        super(ResNet, self).__init__()
        self.include_top = include_top
        self.in_channel = 64

        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, blocks_num[0])             # conv2_x
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)  # conv3_x
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)  # conv4_x
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)  # conv5_x
        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):  #通过遍历模型的所有模块，判断每个模块是否是nn.Conv2d类型的卷积层。如果是，则对该卷积层的权重进行初始化。使用了Kaiming方法（nn.init.kaiming_normal_），其中m.weight表示当前卷积层的权重参数，mode='fan_out'表示权重初始化基于输出通道的数量，nonlinearity='relu'表示使用ReLU作为激活函数。
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    # channel为残差结构中第一层卷积核个数
    def _make_layer(self, block, channel, block_num, stride=1):
        downsample = None

        # ResNet50/101/152的残差结构，block.expansion=4
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))

        layers = []
        layers.append(block(self.in_channel, channel, downsample=downsample, stride=stride))
        self.in_channel = channel * block.expansion  #更新 self.in_channel 为 channel * block.expansion，以便作为下一个残差块的输入通道数。

        for _ in range(1, block_num):
            layers.append(block(self.in_channel, channel))  # layers 中添加剩余的 block_num - 1 个残差块，这些残差块的输入通道数都是 self.in_channel，输出通道数都是 channel

        return nn.Sequential(*layers)  #列表 layers 转换为一个 nn.Sequential 对象并返回，表示当前层的所有残差块按顺序组成的模块。

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.include_top:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)

        return x


def resnet34(num_classes=1000, include_top=True):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)

def resnet50(num_classes=1000, include_top=True):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)


def resnet101(num_classes=1000, include_top=True):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, include_top=include_top)


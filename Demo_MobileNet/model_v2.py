from torch import nn
import torch


def _make_divisible(ch, divisor=8, min_ch=None):
    """
        将输入的通道数(ch)调整到divisor的整数倍，方便硬件加速
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_ch is None:
        min_ch = divisor
    new_ch = max(min_ch, int(ch + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_ch < 0.9 * ch:
        new_ch += divisor
    return new_ch

# 定义普通卷积、BN结构
class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2  # padding的设置根据kernel_size来定，如果kernel_size为3，则padding设置为1；如果kernel_size为1，为padding为0
        super(ConvBNReLU, self).__init__(
            # 在pytorch中，如果设置的 group=1的话，就为普通卷积；如果设置的值为输入特征矩阵的深度的话（即in_channel），则为深度卷积（deptwise conv），并且Dw卷积的输出特征矩阵的深度等于输入特征矩阵的深度
            nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, groups=groups, bias=False),  # groups=1,表示普通的卷积；因为接下来要使用的是BN层，此处的偏置不起任何作用，所以设置为1
            nn.BatchNorm2d(out_channel),
            nn.ReLU6(inplace=True)    # 此处使用的是Relu6激活函数
        )

# 定义mobile网络基本结构--即倒残差结构
class InvertedResidual(nn.Module):
    def __init__(self, in_channel, out_channel, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        hidden_channel = in_channel * expand_ratio
        self.use_shortcut = stride == 1 and in_channel == out_channel  # stride == 1 and in_channel == out_channel：保证输入矩阵与输出矩阵的shape一致，且通道数也一致，这样才可以进行shurtcut

        layers = []
        if expand_ratio != 1:  # 表示如果扩展因子不为1时，则使用1x1的卷积层（即对输入特征矩阵的深度进行扩充）
            # 1x1 pointwise conv
            layers.append(ConvBNReLU(in_channel, hidden_channel, kernel_size=1))
        layers.extend([
            # 3x3 depthwise conv
            # 在pytorch中，如果设置的 group=1的话，就为普通卷积；如果设置的值为输入特征矩阵的深度的话（即in_channel），则为深度卷积（deptwise conv），并且Dw卷积的输出特征矩阵的深度等于输入特征矩阵的深度
            ConvBNReLU(hidden_channel, hidden_channel, stride=stride, groups=hidden_channel),
            # 1x1 pointwise conv(linear)  因为其后跟随的是线性激活函数，即y=x，所以其后面不在跟随激活函数
            nn.Conv2d(hidden_channel, out_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channel),
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_shortcut:
            return x + self.conv(x)
        else:
            return self.conv(x)

# 定义mobileNetV2网络
class MobileNetV2(nn.Module):
    def __init__(self, num_classes=5, alpha=1.0, round_nearest=8):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = _make_divisible(32 * alpha, round_nearest)  # 将卷积核的个数调整为8的整数倍
        last_channel = _make_divisible(1280 * alpha, round_nearest)

        inverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        features = []
        # conv1 layer
        features.append(ConvBNReLU(3, input_channel, stride=2))   # 添加第一层普通卷积层
        # building inverted residual residual blockes
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * alpha, round_nearest)   # 根据alpha因子调整卷积核的个数
            for i in range(n):   # 循环添加倒残差模块
                stride = s if i == 0 else 1  # s表示的是倒残差模块结构中第一层卷积对应的步距，剩余层都是1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t))  # 添加一系列倒残差结构
                input_channel = output_channel
        # building last several layers
        features.append(ConvBNReLU(input_channel, last_channel, 1))  # 构建最后一层卷积层
        # combine feature layers
        self.features = nn.Sequential(*features)

        # building classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # 采用自适应平均采样层
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(last_channel, num_classes)
        )

        # weight initialization  
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)   # 初始化为正态分布的函数，均值为0，方差为0.01
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


if __name__ == '__main__':
    divisible = _make_divisible(1)
    print(divisible)


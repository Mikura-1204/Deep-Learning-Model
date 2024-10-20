from typing import Callable, List, Optional

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from functools import partial

# 得到同传入数据最近的8的整数倍数值
# ciallo: this fuc is used to ensure that channels of all layers can be easily calcualte, for example 224/8=28,
#         so choose 8, たぶん
#         and other reason is adopting the request of hardware acceleration
def _make_divisible(ch, divisor=8, min_ch=None):
    """
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

# 普通卷积、BN、激活层模块
class ConvBNActivation(nn.Sequential):
    def __init__(self,
                 in_planes: int,   # 输入特征矩阵的通道
                 out_planes: int,  # 输出特征矩阵的通道
                 kernel_size: int = 3,
                 stride: int = 1,
                 groups: int = 1,
                 norm_layer: Optional[Callable[..., nn.Module]] = None,   # 在卷积后的BN层
                 activation_layer: Optional[Callable[..., nn.Module]] = None):  # 激活函数
        padding = (kernel_size - 1) // 2
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if activation_layer is None:
            activation_layer = nn.ReLU6
        super(ConvBNActivation, self).__init__(nn.Conv2d(in_channels=in_planes,
                                                         out_channels=out_planes,
                                                         kernel_size=kernel_size,
                                                         stride=stride,
                                                         padding=padding,
                                                         groups=groups,
                                                         bias=False),
                                               norm_layer(out_planes),   # BN层
                                               activation_layer(inplace=True))

# ciallo: this SE mode is just use of adding a weight of all of channels to improve expression ability of feature
# 注意力机制模块（SE模块，即两个全连接层）   
# 该模块的基本流程是：先进行自适应平均池化(1x1)———>1x1的卷积层———>relu激活层———>1x1的卷积池化———>hardsigmoid()激活函数激活
class SqueezeExcitation(nn.Module):
    # ciallo: in my fucking opinion this just attribute a weigh of channels through a FC layer
    def __init__(self, input_c: int, squeeze_factor: int = 4):
        super(SqueezeExcitation, self).__init__()
        squeeze_c = _make_divisible(input_c // squeeze_factor, 8)    # 获得距离input channel最近的8的整数倍的数字
        self.fc1 = nn.Conv2d(input_c, squeeze_c, 1)    # 该卷积的输出的squeeze_c是输入input_c的1/4  其作用与全连接层一样
        self.fc2 = nn.Conv2d(squeeze_c, input_c, 1)

    def forward(self, x: Tensor) -> Tensor:
        # ciallo: 输入特征矩阵 x 进行全局平均池化，将每个通道的特征压缩成一个单一的数值，结果是一个形状为 (batch_size, input_c, 1, 1) 的张量
        scale = F.adaptive_avg_pool2d(x, output_size=(1, 1))   # 将特征矩阵每一个channel上的数据给平均池化到1x1的大小
        scale = self.fc1(scale) # 通过第一个1x1卷积层（相当于全连接层）减少通道数。
        scale = F.relu(scale, inplace=True)
        scale = self.fc2(scale) # 通过第二个1x1卷积层（相当于全连接层）恢复通道数。
        scale = F.hardsigmoid(scale, inplace=True)   # 激活函数
        return scale * x   # 将得到的数据与传入的对应channel数据进行相乘


# ciallo: 複雑の計算 always using a individual class and save
class InvertedResidualConfig:
    def __init__(self,
                 input_c: int,  # block模块中的第一个1x1卷积层的输入channel数
                 kernel: int,   # depthwise卷积的卷积核大小
                 expanded_c: int,   # block模块中的第一个1x1卷积层的输出channel数
                 out_c: int,  # 经过block模块中第二个1x1卷积层处理过后得到的channel数
                 use_se: bool,  # 是否使用注意力机制模块
                 activation: str,   # 激活方式
                 stride: int,       # 步长
                 width_multi: float):  # width_multi：调节每个卷积层所使用channel的倍率因子
        self.input_c = self.adjust_channels(input_c, width_multi)
        self.kernel = kernel
        self.expanded_c = self.adjust_channels(expanded_c, width_multi)
        self.out_c = self.adjust_channels(out_c, width_multi)
        self.use_se = use_se
        self.use_hs = activation == "HS"  # whether using h-swish activation
        self.stride = stride
        
    # ciallo: this static method change channal without instantiation
    @staticmethod
    def adjust_channels(channels: int, width_multi: float):
        return _make_divisible(channels * width_multi, 8)



# 定义block模块
# 此为block模块，其包含第一个1x1卷积层、DeptWis卷积层、SE注意力机制层（判断是否需求）、第二个1x1卷积层、激活函数（需要判断是否是非线性激活）
# ciallo: new method of conv
class InvertedResidual(nn.Module):
    def __init__(self,
                 cnf: InvertedResidualConfig,   # cnf:配置类参数
                 norm_layer: Callable[..., nn.Module]):      # norm_layer：# BN层
        super(InvertedResidual, self).__init__()

        # ciallo: stride is limited to just 1 and 2
        if cnf.stride not in [1, 2]:  # 判断某一层的配置文件，其步长是否满足条件
            raise ValueError("illegal stride value.")

        # 判断是否进行短连接
        # ciallo: residual jumping connection just has two conditions
        self.use_res_connect = (cnf.stride == 1 and cnf.input_c == cnf.out_c)  # 只有当步长为1，并且输入通道等于输出通道数

        # ciallo: the consturction layers list is here
        #         sequence is depended by append, upchannel first then conv then SE then downchannel
        layers: List[nn.Module] = []
        activation_layer = nn.Hardswish if cnf.use_hs else nn.ReLU    # 判断当前的激活函数类型

        # expand
        # 判断是否相等，如果相等，则不适用1x1的卷积层增加channel维度；不相等的话，才使用该层进行升维度
        if cnf.expanded_c != cnf.input_c:
            layers.append(ConvBNActivation(cnf.input_c,
                                           cnf.expanded_c,
                                           kernel_size=1,
                                           norm_layer=norm_layer,
                                           activation_layer=activation_layer))

        # depthwise
        layers.append(ConvBNActivation(cnf.expanded_c,
                                       cnf.expanded_c,
                                       kernel_size=cnf.kernel,   # depthwise卷积的卷积核大小
                                       stride=cnf.stride,
                                       groups=cnf.expanded_c,    # 深度DW卷积
                                       norm_layer=norm_layer,   # BN层
                                       activation_layer=activation_layer))

        # 判断是否需要添加SE模块
        if cnf.use_se:
            layers.append(SqueezeExcitation(cnf.expanded_c))

        # project
        layers.append(ConvBNActivation(cnf.expanded_c,
                                       cnf.out_c,
                                       kernel_size=1,
                                       norm_layer=norm_layer,  # BN 层
                                       activation_layer=nn.Identity))   # 此层的activation_layer就是进行里普通的线性激活，没有做任何的处理

        self.block = nn.Sequential(*layers)
        self.out_channels = cnf.out_c
        self.is_strided = cnf.stride > 1

    def forward(self, x: Tensor) -> Tensor:
        result = self.block(x)
        if self.use_res_connect:
            result += x   # 进行shortcut连接

        return result


# MobileNetV3网络结构基础框架：其包括：模型的第一层卷积层———>nx【bneckBlock模块】———>1x1的卷积层———>自适应平均池化层———>全连接层———>全连接层
class MobileNetV3(nn.Module):
    def __init__(self,
                # ciallo: this is a list, every block config is different
                inverted_residual_setting: List[InvertedResidualConfig],           # beneckBlock结构一系列参数列表
                last_channel: int,   # 对应的是倒数第二个全连接层输出节点数  1280
                
                # ciallo: when trans, this arg should be changed to your num of classes
                num_classes: int = 1000,  # 类别个数 this is last channel
                 
                # ciallo: this two codes means that you can choose, if you not choose.
                #         mean while you can also choose your own layers to it
                #         so its a callable object
                block: Optional[Callable[..., nn.Module]] = None,   # InvertedResidual核心模块
                norm_layer: Optional[Callable[..., nn.Module]] = None):
        super(MobileNetV3, self).__init__()

        if not inverted_residual_setting:
            raise ValueError("The inverted_residual_setting should not be empty.")
        elif not (isinstance(inverted_residual_setting, List) and
                  all([isinstance(s, InvertedResidualConfig) for s in inverted_residual_setting])):
            raise TypeError("The inverted_residual_setting should be List[InvertedResidualConfig]")

        if block is None:
            block = InvertedResidual   # block类

        if norm_layer is None:
            norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.01)  # partial()为python方法，即为nn.BatchNorm2d传入默认的两个参数

        layers: List[nn.Module] = []

        # building first layer
        # 构建第一层卷积结构
        firstconv_output_c = inverted_residual_setting[0].input_c   # 表示第一个卷积层输出的channel数
        layers.append(ConvBNActivation(3,   # 输入图像数据的channel数
                                       firstconv_output_c,    # 输出channel
                                       kernel_size=3,
                                       stride=2,
                                       norm_layer=norm_layer,
                                       activation_layer=nn.Hardswish))
        # building inverted residual blocks
        # ciallo: trans message config in list to different block 
        # 利用循环的方式添加block模块，将每层的配置文件传给block
        for cnf in inverted_residual_setting:
            layers.append(block(cnf, norm_layer))

        # building last several layers
        lastconv_input_c = inverted_residual_setting[-1].out_c  # 最后的bneckblock的输出channel
        lastconv_output_c = 6 * lastconv_input_c    # lastconv_output_c 与 最后的bneckblock的输出channel数是六倍的关系

        # 定义最后一层的卷积层
        layers.append(ConvBNActivation(lastconv_input_c,   # 最后的bneckblock的输出channel数
                                       lastconv_output_c,   # lastconv_output_c 与 最后的bneckblock的输出channel数是六倍的关系
                                       kernel_size=1,
                                       norm_layer=norm_layer,
                                       activation_layer=nn.Hardswish))
        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Linear(lastconv_output_c, last_channel),
                                        nn.Hardswish(inplace=True),
                                        nn.Dropout(p=0.2, inplace=True),
                                        nn.Linear(last_channel, num_classes))

        # initial weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)
                
    # ciallo: 通过 features 提取特征, 使用 avgpool 进行池化, 将池化后的特征展平为一维向量, 通过 classifier 进行分类
    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


# ciallo: same to forward, according to your condition change num_class !
#         num_classes：分类任务的类别数，默认为5。
#         reduced_tail：是否减少网络参数，默认为 False。
#         width_multi：宽度乘数，用于调整通道数。
#         bneck_conf：一个简化 InvertedResidualConfig 配置的函数。
#         adjust_channels：一个简化通道调整的函数。
#         reduce_divider：用于减少通道数的除数，如果 reduced_tail 为 True，则设置为2，否则为1。
#         inverted_residual_setting：定义了网络的每一层，包括输入通道数、卷积核大小、扩展通道数、输出通道数、是否使用 SE 模块、激活函数和步长。
#         last_channel：最后一层的通道数，经过 adjust_channels 调整
# 
# ciallo: mobilenet_v3_large 更适用于需要高精度的任务，包含更多的参数和计算量。
#         mobilenet_v3_small 更适用于计算资源有限的设备，参数和计算量较少，但仍保持较好的性能
# 
### 构建large基础mobilenet_v3_large模型
def mobilenet_v3_large(num_classes: int = 5,
                       reduced_tail: bool = False) -> MobileNetV3:
    """
    Constructs a large MobileNetV3 architecture from
    "Searching for MobileNetV3" <https://arxiv.org/abs/1905.02244>.

    weights_link:
    https://download.pytorch.org/models/mobilenet_v3_large-8738ca79.pth

    Args:
        num_classes (int): number of classes
        reduced_tail (bool): If True, reduces the channel counts of all feature layers
            between C4 and C5 by 2. It is used to reduce the channel redundancy in the
            backbone for Detection and Segmentation.
    """
    width_multi = 1.0
    bneck_conf = partial(InvertedResidualConfig, width_multi=width_multi)
    adjust_channels = partial(InvertedResidualConfig.adjust_channels, width_multi=width_multi)

    reduce_divider = 2 if reduced_tail else 1   # 是否减少网络参数标志，默认是False，即不减少

    # beneckBlock结构一系列参数列表
    inverted_residual_setting = [
        # input_c, kernel, expanded_c, out_c, use_se, activation, stride
        bneck_conf(16, 3, 16, 16, False, "RE", 1),
        bneck_conf(16, 3, 64, 24, False, "RE", 2),  # C1
        bneck_conf(24, 3, 72, 24, False, "RE", 1),
        bneck_conf(24, 5, 72, 40, True, "RE", 2),  # C2
        bneck_conf(40, 5, 120, 40, True, "RE", 1),
        bneck_conf(40, 5, 120, 40, True, "RE", 1),
        bneck_conf(40, 3, 240, 80, False, "HS", 2),  # C3
        bneck_conf(80, 3, 200, 80, False, "HS", 1),
        bneck_conf(80, 3, 184, 80, False, "HS", 1),
        bneck_conf(80, 3, 184, 80, False, "HS", 1),
        bneck_conf(80, 3, 480, 112, True, "HS", 1),
        bneck_conf(112, 3, 672, 112, True, "HS", 1),
        bneck_conf(112, 5, 672, 160 // reduce_divider, True, "HS", 2),  # C4
        bneck_conf(160 // reduce_divider, 5, 960 // reduce_divider, 160 // reduce_divider, True, "HS", 1),
        bneck_conf(160 // reduce_divider, 5, 960 // reduce_divider, 160 // reduce_divider, True, "HS", 1),
    ]
    last_channel = adjust_channels(1280 // reduce_divider)  # C5

    return MobileNetV3(inverted_residual_setting=inverted_residual_setting,
                       last_channel=last_channel,
                       num_classes=num_classes)

### 构建small基础mobilenet_v3_small模型
def mobilenet_v3_small(num_classes: int = 1000,
                       reduced_tail: bool = False) -> MobileNetV3:
    """
    Constructs a large MobileNetV3 architecture from
    "Searching for MobileNetV3" <https://arxiv.org/abs/1905.02244>.

    weights_link:
    https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth

    Args:
        num_classes (int): number of classes
        reduced_tail (bool): If True, reduces the channel counts of all feature layers
            between C4 and C5 by 2. It is used to reduce the channel redundancy in the
            backbone for Detection and Segmentation.
    """
    width_multi = 1.0
    bneck_conf = partial(InvertedResidualConfig, width_multi=width_multi)
    adjust_channels = partial(InvertedResidualConfig.adjust_channels, width_multi=width_multi)

    reduce_divider = 2 if reduced_tail else 1

    inverted_residual_setting = [
        # input_c, kernel, expanded_c, out_c, use_se, activation, stride
        bneck_conf(16, 3, 16, 16, True, "RE", 2),  # C1
        bneck_conf(16, 3, 72, 24, False, "RE", 2),  # C2
        bneck_conf(24, 3, 88, 24, False, "RE", 1),
        bneck_conf(24, 5, 96, 40, True, "HS", 2),  # C3
        bneck_conf(40, 5, 240, 40, True, "HS", 1),
        bneck_conf(40, 5, 240, 40, True, "HS", 1),
        bneck_conf(40, 5, 120, 48, True, "HS", 1),
        bneck_conf(48, 5, 144, 48, True, "HS", 1),
        bneck_conf(48, 5, 288, 96 // reduce_divider, True, "HS", 2),  # C4
        bneck_conf(96 // reduce_divider, 5, 576 // reduce_divider, 96 // reduce_divider, True, "HS", 1),
        bneck_conf(96 // reduce_divider, 5, 576 // reduce_divider, 96 // reduce_divider, True, "HS", 1)
    ]
    last_channel = adjust_channels(1024 // reduce_divider)  # C5

    return MobileNetV3(inverted_residual_setting=inverted_residual_setting,
                       last_channel=last_channel,
                       num_classes=num_classes)
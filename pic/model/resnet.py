from torch import nn
from pic.model.register import register_model


class ConvNormAct(nn.Sequential):
    def __init__(self, in_ch, out_ch, kernel_size, norm_layer=nn.BatchNorm2d, stride=1, padding=0, groups=1, act=True):
        super(ConvNormAct, self).__init__(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=False, groups=groups),
            norm_layer(out_ch),
            nn.ReLU(inplace=True) if act else nn.Identity()
        )


class SEUnit(nn.Sequential):
    def __init__(self, ch, norm_layer, r=16):
        super(SEUnit, self).__init__(
            nn.AdaptiveAvgPool2d(1), # squeeze
            ConvNormAct(ch, ch//r, 1, norm_layer), nn.Conv2d(ch//r, ch, 1, bias=True), nn.Sigmoid(), # excitation
        )
    def forward(self, x):
        out = super(SEUnit, self).forward(x)
        return out * x

class BasicBlock(nn.Module):
    factor = 1
    def __init__(self, in_channels, out_channels, stride, norm_layer, downsample=None, groups=1, base_width=64,
                 drop_path_rate=0.0, se=False):
        super(BasicBlock, self).__init__()
        self.conv1 = ConvNormAct(in_channels, out_channels, 3, norm_layer, stride, 1)
        self.conv2 = ConvNormAct(out_channels, out_channels, 3, norm_layer, 1, 1, act=False)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample if downsample else nn.Identity()
        self.drop_path = StochasticDepth(drop_path_rate)
        self.se = SEUnit(out_channels, norm_layer) if se else nn.Identity()

    def forward(self, x):
        out = self.conv1(x)
        return self.relu(self.downsample(x) + self.drop_path(self.se(self.conv2(out))))


class BottleNeck(nn.Module):
    factor = 4
    def __init__(self, in_channels, out_channels, stride, norm_layer, downsample=None, groups=1, base_width=64,
                 drop_path_rate=0.0, se=False):
        super(BottleNeck, self).__init__()
        self.width = width = int(out_channels * (base_width / 64.0)) * groups
        self.out_channels = out_channels * self.factor
        self.conv1 = ConvNormAct(in_channels, width, 1, norm_layer)
        self.conv2 = ConvNormAct(width, width, 3, norm_layer, stride, 1, groups=groups)
        self.conv3 = ConvNormAct(width, self.out_channels, 1, norm_layer, act=False)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample if downsample else nn.Identity()
        self.drop_path = StochasticDepth(drop_path_rate)
        self.se = SEUnit(self.out_channels, norm_layer) if se else nn.Identity()

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        return self.relu(self.downsample(x) + self.drop_path(self.se(self.conv3(out))))


class StochasticDepth(nn.Module):
    def __init__(self, prob, mode='row'):
        super(StochasticDepth, self).__init__()
        self.prob = prob
        self.survival = 1.0 - prob
        self.mode = mode

    def forward(self, x):
        if self.prob == 0.0 or not self.training:
            return x
        else:
            shape = [x.size(0)] + [1] * (x.ndim - 1) if self.mode == 'row' else [1]
            return x * x.new_empty(shape).bernoulli_(self.survival).div_(self.survival)


model_config = {
    # resnet
    'resnet34': {'parameter':dict(nblock=[3, 4, 6, 3], block=BasicBlock), 'etc':{}},
    'resnet50': {'parameter':dict(nblock=[3, 4, 6, 3], block=BottleNeck), 'etc':{}},
    'resnet101': {'parameter': dict(nblock=[3, 4, 23, 3], block=BottleNeck), 'etc': {}},
    'resnet152': {'parameter': dict(nblock=[3, 8, 36, 3], block=BottleNeck), 'etc': {}},

    # resnext
    'resnext50_32_4': {'parameter': dict(nblock=[3, 4, 6, 3], groups=32, base_width=4, block=BottleNeck), 'etc': {}},
    'resnext101_32_4': {'parameter': dict(nblock=[3, 4, 23, 3], groups=32, base_width=4, block=BottleNeck), 'etc': {}},
    'resnext152_32_4': {'parameter': dict(nblock=[3, 8, 36, 3], groups=32, base_width=4, block=BottleNeck), 'etc': {}},

    # se-resnet
    'seresnet34': {'parameter': dict(nblock=[3, 4, 6, 3], block=BasicBlock, se=True), 'etc': {}},
    'seresnet50': {'parameter': dict(nblock=[3, 4, 6, 3], block=BottleNeck, se=True), 'etc': {}},
    'seresnet101': {'parameter': dict(nblock=[3, 4, 23, 3], block=BottleNeck, se=True), 'etc': {}},
    'seresnet152': {'parameter': dict(nblock=[3, 8, 36, 3], block=BottleNeck, se=True), 'etc': {}},

    # seresnext
    'seresnext50_32_4': {'parameter': dict(nblock=[3, 4, 6, 3], groups=32, base_width=4, block=BottleNeck, se=True), 'etc': {}},
    'seresnext101_32_4': {'parameter': dict(nblock=[3, 4, 23, 3], groups=32, base_width=4, block=BottleNeck, se=True), 'etc': {}},
    'seresnext152_32_4': {'parameter': dict(nblock=[3, 8, 36, 3], groups=32, base_width=4, block=BottleNeck, se=True), 'etc': {}},
}


@register_model
class ResNet(nn.Module):
    def __init__(self,
                 nblock,
                 block = BottleNeck,
                 norm_layer: nn.Module = nn.BatchNorm2d,
                 channels=[64, 128, 256, 512],
                 strides=[1, 2, 2, 2],
                 groups=1,
                 base_width=64,
                 zero_init_last=True,
                 num_classes=1000,
                 in_channels=3,
                 drop_path_rate=0.0,
                 se=False) -> None:
        super(ResNet, self).__init__()
        self.groups = groups
        self.num_classes = num_classes
        self.base_width = base_width
        self.norm_layer = norm_layer
        self.in_channels = channels[0]
        self.out_channels = channels[-1] * block.factor
        self.num_block = sum(nblock)
        self.cur_block = 0
        self.drop_path_rate = drop_path_rate
        self.se = se

        self.conv1 = nn.Conv2d(in_channels, self.in_channels, kernel_size=(7, 7), stride=2, padding=(3, 3), bias=False)
        self.bn1 = self.norm_layer(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=1)

        self.layers = [self.make_layer(block=block, nblock=nblock[i], channels=channels[i], stride=strides[i]) for i in range(len(nblock))]
        self.flatten = nn.Flatten()
        self.classifier = nn.Linear(self.out_channels, self.num_classes)
        self.last_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.register_layer()
        self.init_weight(zero_init_last)

    def register_layer(self):
        for i, layer in enumerate(self.layers):
            exec('self.layer{} = {}'.format(i + 1, 'layer'))

    def get_drop_path_rate(self):
        drop_path_rate = self.drop_path_rate * (self.cur_block / self.num_block)
        self.cur_block += 1
        return drop_path_rate

    def make_layer(self, block, nblock: int, channels: int, stride: int) -> nn.Sequential:
        if self.in_channels != channels * block.factor or stride != 1:
            downsample = ConvNormAct(self.in_channels, channels * block.factor, 1, self.norm_layer, stride, act=False)
        else:
            downsample = None

        layers = []
        for i in range(nblock):
            if i == 1:
                stride = 1
                downsample = None
                self.in_channels = channels * block.factor
            layers.append(block(in_channels=self.in_channels, out_channels=channels, stride=stride,
                                norm_layer=self.norm_layer, downsample=downsample, groups=self.groups,
                                base_width=self.base_width, drop_path_rate=self.get_drop_path_rate(), se=self.se))
        return nn.Sequential(*layers)

    def forward(self, x):
        stem = self.relu(self.bn1(self.conv1(x)))
        x = self.max_pool(stem)

        for layer in self.layers:
            x = layer(x)

        x = self.flatten(self.last_pool(x))

        return self.classifier(x)

    def init_weight(self, zero_init_last=True):
        for m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_last:
            for m in self.named_modules():
                if isinstance(m, BottleNeck):
                    nn.init.constant_(m.bn3.weight, 0)



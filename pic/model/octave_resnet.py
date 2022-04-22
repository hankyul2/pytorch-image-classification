from torch import nn
import torch.nn.functional as F
from pic.model.register import register_model


class ConvNormAct(nn.Sequential):
    def __init__(self, in_ch, out_ch, kernel_size, norm_layer=nn.BatchNorm2d, stride=1, padding=0, groups=1, act=True):
        super(ConvNormAct, self).__init__(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=False, groups=groups),
            norm_layer(out_ch),
            nn.ReLU(inplace=True) if act else nn.Identity()
        )


class SEUnit(nn.Sequential):
    def __init__(self, ch, norm_layer=nn.BatchNorm2d, r=16):
        super(SEUnit, self).__init__(
            nn.AdaptiveAvgPool2d(1), # squeeze
            ConvNormAct(ch, ch//r, 1, norm_layer), nn.Conv2d(ch//r, ch, 1, bias=True), nn.Sigmoid(), # excitation
        )
    def forward(self, x):
        out = super(SEUnit, self).forward(x)
        return out * x


class HighToHigh(nn.Module):
    def __init__(self, in_ch, out_ch, in_alpha, out_alpha, kernel_size, padding, groups, act, se):
        super(HighToHigh, self).__init__()
        self.conv_high_to_high = ConvNormAct(in_ch, out_ch, kernel_size, stride=1, padding=padding, groups=groups, act=act)
        self.se_high = SEUnit(out_ch) if se else nn.Identity()

    def forward(self, h, l=None):
        h_out = self.se_high(self.conv_high_to_high(h))
        l_out = l
        return h_out, l_out


class HighToHighLow(nn.Module):
    def __init__(self, in_ch, out_ch, in_alpha, out_alpha, kernel_size, padding, groups, act):
        super(HighToHighLow, self).__init__()
        self.down_sample = nn.AvgPool2d((2,2))
        self.conv_high_to_high = ConvNormAct(in_ch, int(out_ch * (1 - out_alpha)), kernel_size, stride=1, padding=padding, groups=groups, act=act)
        self.conv_high_to_low = ConvNormAct(in_ch, int(out_ch * out_alpha), kernel_size, stride=1, padding=padding, groups=groups, act=act)

    def forward(self, h, l=None):
        h_out = self.conv_high_to_high(h)
        l_out = self.conv_high_to_low(self.down_sample(h))
        return h_out, l_out


class HighLowToHigh(nn.Module):
    def __init__(self, in_ch, out_ch, in_alpha, out_alpha, kernel_size, padding, groups, act):
        super(HighLowToHigh, self).__init__()
        self.conv_high_to_high = ConvNormAct(int(in_ch * (1 - in_alpha)), out_ch, kernel_size, stride=1, padding=padding, groups=groups, act=False)
        self.conv_low_to_high = ConvNormAct(int(in_ch * in_alpha), out_ch, kernel_size, stride=1, padding=padding, groups=groups, act=False)
        self.act = nn.ReLU(inplace=True) if act else nn.Identity()

    def forward(self, h, l):
        h_out = self.act(self.conv_high_to_high(h) + F.interpolate(self.conv_low_to_high(l), h.shape[-2:]))
        l_out = None
        return h_out, l_out


class HighLowToHighLow(nn.Module):
    def __init__(self, in_ch, out_ch, in_alpha, out_alpha, kernel_size, padding, groups, act, se):
        super(HighLowToHighLow, self).__init__()
        self.up_sample = nn.Upsample(scale_factor=2, mode='nearest')
        self.down_sample = nn.AvgPool2d((2,2))
        self.conv_high_to_high = ConvNormAct(int(in_ch * (1 - in_alpha)), int(out_ch * (1 - out_alpha)), kernel_size, stride=1, padding=padding, groups=groups, act=False)
        self.conv_high_to_low = ConvNormAct(int(in_ch * (1 - in_alpha)), int(out_ch * out_alpha), kernel_size, stride=1, padding=padding, groups=groups, act=False)
        self.conv_low_to_high = ConvNormAct(int(in_ch * in_alpha), int(out_ch * (1 - out_alpha)), kernel_size, stride=1, padding=padding, groups=groups, act=False)
        self.conv_low_to_low = ConvNormAct(int(in_ch * in_alpha), int(out_ch * out_alpha), kernel_size, stride=1, padding=padding, groups=groups, act=False)
        self.act = nn.ReLU(inplace=True) if act else nn.Identity()
        self.se_high = SEUnit(int(out_ch * (1 - out_alpha))) if se else nn.Identity()
        self.se_low = SEUnit(int(out_ch * out_alpha)) if se else nn.Identity()

    def forward(self, h, l):
        h_out = self.se_high(self.act(self.conv_high_to_high(h) + self.up_sample(self.conv_low_to_high(l))))
        l_out = self.se_low(self.act(self.conv_low_to_low(l) + self.conv_high_to_low(self.down_sample(h))))
        return h_out, l_out


class OctaveConvNormAct(nn.Module):
    def __init__(self, in_ch, out_ch, in_alpha, out_alpha, kernel_size=1, stride=1, padding=0, groups=1, act=True, se=False):
        super(OctaveConvNormAct, self).__init__()
        self.down_sample = nn.AvgPool2d((2,2))
        self.up_sample = nn.Upsample(scale_factor=2, mode='nearest')
        self.stride = stride

        # stage 4, second ~
        if in_alpha == 0 and out_alpha == 0:
            block = HighToHigh(in_ch, out_ch, in_alpha, out_alpha, kernel_size, padding, groups, act, se)
        # stage 1, first
        elif in_alpha == 0:
            block = HighToHighLow(in_ch, out_ch, in_alpha, out_alpha, kernel_size, padding, groups, act)
        # stage 4, first
        elif out_alpha == 0:
            block = HighLowToHigh(in_ch, out_ch, in_alpha, out_alpha, kernel_size, padding, groups, act)
        # stage 1, second ~ 3, last
        else:
            block = HighLowToHighLow(in_ch, out_ch, in_alpha, out_alpha, kernel_size, padding, groups, act, se)
        self.conv = block

    def forward(self, x):
        high, low = x if type(x) is tuple else (x, None)

        if self.stride == 2:
            high = self.down_sample(high)
            low = self.down_sample(low) if low is not None else low

        return self.conv(high, low)


class OctBasicBlock(nn.Module):
    factor = 1
    def __init__(self, in_channels, out_channels, in_alpha, out_alpha, stride, norm_layer, downsample=None, groups=1, base_width=64,
                 drop_path_rate=0.0, se=False):
        super(OctBasicBlock, self).__init__()
        self.conv1 = OctaveConvNormAct(in_channels, out_channels, in_alpha, out_alpha, 3, stride, 1)
        self.conv2 = OctaveConvNormAct(out_channels, out_channels, in_alpha, out_alpha, 3, 1, 1, act=False, se=se)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample if downsample else nn.Identity()
        self.drop_path = StochasticDepth(drop_path_rate)

    def forward(self, x):
        out = self.conv1(x)
        h, l = self.conv2(out)
        h_r, l_r = self.downsample(x)

        h = self.relu(self.drop_path(h) + h_r)
        l = self.relu(self.drop_path(l) + l_r) if l is not None else None

        return h, l


class OctBottleNeck(nn.Module):
    factor = 4
    def __init__(self, in_channels, out_channels, in_alpha, out_alpha, stride, norm_layer, downsample=None, groups=1, base_width=64,
                 drop_path_rate=0.0, se=False):
        super(OctBottleNeck, self).__init__()
        self.width = width = int(out_channels * (base_width / 64.0)) * groups
        self.out_channels = out_channels * self.factor
        self.conv1 = OctaveConvNormAct(in_channels, width, in_alpha, out_alpha, 1)
        self.conv2 = OctaveConvNormAct(width, width, out_alpha, out_alpha, 3, stride, 1, groups)
        self.conv3 = OctaveConvNormAct(width, self.out_channels, out_alpha, out_alpha, 1, act=False, se=se)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample if downsample else nn.Identity()
        self.drop_path = StochasticDepth(drop_path_rate)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        h, l = self.conv3(out)
        h_r, l_r = self.downsample(x)
        
        h = self.relu(self.drop_path(h) + h_r)
        l = self.relu(self.drop_path(l) + l_r) if l is not None else None

        return h, l


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
    'oct_resnet34_alpha5': {'parameter':dict(nblock=[3, 4, 6, 3], alpha=0.5, block=OctBasicBlock), 'etc':{}},
    'oct_resnet50_alpha5': {'parameter':dict(nblock=[3, 4, 6, 3], alpha=0.5, block=OctBottleNeck), 'etc':{}},
    'oct_resnet101_alpha5': {'parameter': dict(nblock=[3, 4, 23, 3], alpha=0.5, block=OctBottleNeck), 'etc': {}},
    'oct_resnet152_alpha5': {'parameter': dict(nblock=[3, 8, 36, 3], alpha=0.5, block=OctBottleNeck), 'etc': {}},

    # resnext
    'oct_resnext50_32_4_alpha5': {'parameter': dict(nblock=[3, 4, 6, 3], groups=32, base_width=4, block=OctBottleNeck), 'etc': {}},
    'oct_resnext101_32_4_alpha5': {'parameter': dict(nblock=[3, 4, 23, 3], groups=32, base_width=4, block=OctBottleNeck), 'etc': {}},
    'oct_resnext152_32_4_alpha5': {'parameter': dict(nblock=[3, 8, 36, 3], groups=32, base_width=4, block=OctBottleNeck), 'etc': {}},

    # se-resnet
    'oct_seresnet34_alpha5': {'parameter': dict(nblock=[3, 4, 6, 3], alpha=0.5, block=OctBasicBlock, se=True), 'etc': {}},
    'oct_seresnet50_alpha5': {'parameter': dict(nblock=[3, 4, 6, 3], alpha=0.5, block=OctBottleNeck, se=True), 'etc': {}},
    'oct_seresnet101_alpha5': {'parameter': dict(nblock=[3, 4, 23, 3], alpha=0.5, block=OctBottleNeck, se=True), 'etc': {}},
    'oct_seresnet152_alpha5': {'parameter': dict(nblock=[3, 8, 36, 3], alpha=0.5, block=OctBottleNeck, se=True), 'etc': {}},

    # seresnext
    'oct_seresnext50_32_4_alpha5': {'parameter': dict(nblock=[3, 4, 6, 3], groups=32, base_width=4, block=OctBottleNeck, se=True), 'etc': {}},
    'oct_seresnext101_32_4_alpha5': {'parameter': dict(nblock=[3, 4, 23, 3], groups=32, base_width=4, block=OctBottleNeck, se=True), 'etc': {}},
    'oct_seresnext152_32_4_alpha5': {'parameter': dict(nblock=[3, 8, 36, 3], groups=32, base_width=4, block=OctBottleNeck, se=True), 'etc': {}},
}


@register_model
class OctResNet(nn.Module):
    def __init__(self,
                 nblock,
                 block = OctBottleNeck,
                 norm_layer: nn.Module = nn.BatchNorm2d,
                 channels=[64, 128, 256, 512],
                 strides=[1, 2, 2, 2],
                 alpha=0.15,
                 groups=1,
                 base_width=64,
                 zero_init_last=True,
                 num_classes=1000,
                 in_channels=3,
                 drop_path_rate=0.0,
                 se=False) -> None:
        super(OctResNet, self).__init__()
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

        alpha = [(0, alpha), (alpha, alpha), (alpha, alpha), (alpha, 0)]
        self.layers = [self.make_layer(block=block, nblock=nblock[i], channels=channels[i], stride=strides[i], alpha=alpha[i]) for i in range(len(nblock))]
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

    def make_layer(self, block, nblock: int, channels: int, stride: int, alpha: tuple) -> nn.Sequential:
        in_alpha, out_alpha = alpha
        if self.in_channels != channels * block.factor or stride != 1:
            downsample = OctaveConvNormAct(self.in_channels, channels * block.factor, in_alpha, out_alpha, 1, stride, act=False)
        else:
            downsample = None

        layers = []
        for i in range(nblock):
            if i == 1:
                stride = 1
                downsample = None
                in_alpha = out_alpha
                self.in_channels = channels * block.factor
            layers.append(block(in_channels=self.in_channels, out_channels=channels, stride=stride,
                                norm_layer=self.norm_layer, downsample=downsample, groups=self.groups,
                                base_width=self.base_width, drop_path_rate=self.get_drop_path_rate(),
                                in_alpha=in_alpha, out_alpha=out_alpha, se=self.se))
        return nn.Sequential(*layers)

    def forward(self, x):
        stem = self.relu(self.bn1(self.conv1(x)))
        x = self.max_pool(stem)

        for layer in self.layers:
            x = layer(x)

        x = self.flatten(self.last_pool(x[0]))

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
                if isinstance(m, OctBottleNeck):
                    nn.init.constant_(m.weight, 0)



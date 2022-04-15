import torch
from torch import nn
import torch.nn.functional as F
from pic.model.register import register_model


class ConvNormAct(nn.Sequential):
    def __init__(self, in_ch, out_ch, kernel_size, norm_layer=nn.BatchNorm2d, stride=1, padding=0, groups=1, dilation=1, act=True):
        super(ConvNormAct, self).__init__(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, dilation=dilation, bias=False, groups=groups),
            norm_layer(out_ch),
            nn.ReLU(inplace=True) if act else nn.Identity()
        )


class SelectiveKernel(nn.Module):
    def __init__(self, width, groups, stride, norm_layer, version='original', m=2, r=16):
        super(SelectiveKernel, self).__init__()
        reduced_dim = max(width//r, 32)
        self.m = m
        self.width = width
        self.version = version
        self.fuse = nn.Sequential(nn.AdaptiveAvgPool2d(1), ConvNormAct(width, reduced_dim, 1, norm_layer))
        self.select = nn.Conv2d(reduced_dim, width * m, 1, 1, 0)

        if version == 'original':
            self.split = nn.ModuleList([ConvNormAct(width, width, 3, norm_layer, stride, dilation, groups, dilation) for dilation in range(1, 1+m) ])
        elif version == 'timm':
            self.split = nn.ModuleList([ConvNormAct(width//m, width, 3, norm_layer, stride, dilation, groups, dilation) for dilation in range(1, 1+m) ])
        elif version == 'ensemble':
            self.split = ConvNormAct(width, width * m, 3, norm_layer, stride, 1, groups)

    def forward(self, x):
        if self.version == 'original':
            features = [conv(x) for conv in self.split]
        elif self.version == 'timm':
            xs = torch.split(x, self.width // self.m, 1)
            features = [conv(x) for x, conv in zip(xs, self.split)]
        elif self.version == 'ensemble':
            features = torch.split(self.split(x), self.width, 1)

        features = torch.stack(features, dim=1)
        z = self.fuse(torch.sum(features, dim=1))
        attn = self.select(z).reshape(-1, self.m, self.width, 1, 1)
        attn = F.softmax(attn, dim=1)
        return torch.sum(features * attn, dim=1)


class BottleNeck(nn.Module):
    factor = 4
    def __init__(self, in_channels, out_channels, stride, norm_layer, downsample=None, groups=1, base_width=64,
                 drop_path_rate=0.0, version='original'):
        super(BottleNeck, self).__init__()
        self.width = width = int(out_channels * (base_width / 64.0)) * groups
        self.out_channels = out_channels * self.factor
        self.conv1 = ConvNormAct(in_channels, width, 1, norm_layer)
        self.conv2 = SelectiveKernel(width, groups, stride, norm_layer, version=version)
        self.conv3 = ConvNormAct(width, self.out_channels, 1, norm_layer, act=False)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample if downsample else nn.Identity()
        self.drop_path = StochasticDepth(drop_path_rate)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        return self.relu(self.downsample(x) + self.drop_path(self.conv3(out)))


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
    'sknet26': {'parameter': dict(nblock=[2, 2, 2, 2], version='original', block=BottleNeck), 'etc': {}},
    'sknet50': {'parameter': dict(nblock=[3, 4, 6, 3], version='original', block=BottleNeck), 'etc': {}},
    'sknet101': {'parameter': dict(nblock=[3, 4, 23, 3], version='original', block=BottleNeck), 'etc': {}},

    # resnet - timm
    'sknet26_timm': {'parameter': dict(nblock=[2, 2, 2, 2], version='timm', block=BottleNeck), 'etc': {}},
    'sknet50_timm': {'parameter': dict(nblock=[3, 4, 6, 3], version='timm', block=BottleNeck), 'etc': {}},
    'sknet101_timm': {'parameter': dict(nblock=[3, 4, 23, 3], version='timm', block=BottleNeck), 'etc': {}},

    # resnet - ensemble
    'sknet26_ensemble': {'parameter': dict(nblock=[2, 2, 2, 2], version='ensemble', block=BottleNeck), 'etc': {}},
    'sknet50_ensemble': {'parameter': dict(nblock=[3, 4, 6, 3], version='ensemble', block=BottleNeck), 'etc': {}},
    'sknet101_ensemble': {'parameter': dict(nblock=[3, 4, 23, 3], version='ensemble', block=BottleNeck), 'etc': {}},

    # resnext
    'sknet26_32_4': {'parameter': dict(nblock=[2, 2, 2, 2], groups=32, base_width=4, block=BottleNeck), 'etc': {}},
    'sknet50_32_4': {'parameter': dict(nblock=[3, 4, 6, 3], groups=32, base_width=4, block=BottleNeck), 'etc': {}},
    'sknet101_32_4': {'parameter': dict(nblock=[3, 4, 23, 3], groups=32, base_width=4, block=BottleNeck), 'etc': {}},
}


@register_model
class SKNet(nn.Module):
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
                 version='original') -> None:
        super(SKNet, self).__init__()
        self.groups = groups
        self.num_classes = num_classes
        self.base_width = base_width
        self.norm_layer = norm_layer
        self.in_channels = channels[0]
        self.out_channels = channels[-1] * block.factor
        self.num_block = sum(nblock)
        self.cur_block = 0
        self.drop_path_rate = drop_path_rate
        self.version = version

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
                                base_width=self.base_width, drop_path_rate=self.get_drop_path_rate(),
                                version=self.version,))

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



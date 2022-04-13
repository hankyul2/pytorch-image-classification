import torch
from torch import nn
from pic.model.register import register_model


class ConvNormAct(nn.Sequential):
    def __init__(self, in_ch, out_ch, kernel_size, norm_layer=nn.BatchNorm2d, stride=1, padding=0, groups=1, act=True):
        super(ConvNormAct, self).__init__(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=False, groups=groups),
            norm_layer(out_ch),
            nn.ReLU(inplace=True) if act else nn.Identity()
        )


class BasicBlock(nn.Module):
    factor = 1
    def __init__(self, in_channels, out_channels, stride, norm_layer, downsample=None, groups=1, base_width=64,
                 scale=4, drop_path_rate=0.0):
        super(BasicBlock, self).__init__()
        self.scale = scale
        self.stride = stride
        self.width = width = int(out_channels * (base_width / 64.0)) * groups
        self.conv1 = ConvNormAct(in_channels, width * scale, 3, norm_layer, 1, 1)
        self.conv2 = nn.ModuleList([ConvNormAct(width, width, 3, norm_layer, stride, 1, groups) for _ in range(scale - 1)])
        self.conv3 = ConvNormAct(width*scale, out_channels * self.factor, 1, norm_layer, act=False)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample if downsample else nn.Identity()
        self.avg_pool = nn.AvgPool2d(kernel_size=3, stride=stride, padding=1) if stride != 1 else nn.Identity()
        self.drop_path = StochasticDepth(drop_path_rate)

    def forward(self, x):
        out = self.conv1(x)

        in_groups, out_groups = torch.split(out, self.width, 1), []
        for group_id, (in_group, layer) in enumerate(zip(in_groups, self.conv2)):
            if group_id != 0 and self.stride != 2:
                in_group = in_group + out_group
            out_group = layer(in_group)
            out_groups.append(out_group)
        out_groups.append(self.avg_pool(in_groups[self.scale-1]))
        out = torch.cat(out_groups, dim=1)

        return self.relu(self.downsample(x) + self.drop_path(self.conv3(out)))


class BottleNeck(nn.Module):
    factor = 4
    def __init__(self, in_channels, out_channels, stride, norm_layer, downsample=None, groups=1, base_width=64,
                 scale=4, drop_path_rate=0.0):
        super(BottleNeck, self).__init__()
        self.scale = scale
        self.stride = stride
        self.width = width = int(out_channels * (base_width / 64.0)) * groups
        self.out_channels = out_channels * self.factor
        self.conv1 = ConvNormAct(in_channels, width * scale, 1, norm_layer)
        self.conv2 = nn.ModuleList([ConvNormAct(width, width, 3, norm_layer, stride, 1, groups) for _ in range(scale - 1)])
        self.conv3 = ConvNormAct(width*scale, out_channels * self.factor, 1, norm_layer, act=False)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample if downsample else nn.Identity()
        self.avg_pool = nn.AvgPool2d(kernel_size=3, stride=stride, padding=1) if stride != 1 else nn.Identity()
        self.drop_path = StochasticDepth(drop_path_rate)

    def forward(self, x):
        out = self.conv1(x)

        in_groups, out_groups = torch.split(out, self.width, 1), []
        for group_id, (in_group, layer) in enumerate(zip(in_groups, self.conv2)):
            if group_id != 0 and self.stride != 2:
                in_group = in_group + out_group
            out_group = layer(in_group)
            out_groups.append(out_group)
        out_groups.append(self.avg_pool(in_groups[self.scale-1]))
        out = torch.cat(out_groups, dim=1)

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
    # This 18w_4s model does not converge
    'res2net34_18w_4s': {'parameter':dict(nblock=[3, 4, 6, 3], base_width=18, scale=4, block=BasicBlock), 'etc':{}},
    'res2net50_18w_4s': {'parameter':dict(nblock=[3, 4, 6, 3], base_width=18, scale=4, block=BottleNeck), 'etc':{}},
    'res2net101_18w_4s': {'parameter': dict(nblock=[3, 4, 23, 3], base_width=18, scale=4, block=BottleNeck), 'etc': {}},
    'res2net152_18w_4s': {'parameter': dict(nblock=[3, 8, 36, 3], base_width=18, scale=4, block=BottleNeck), 'etc': {}},

    # Res2Net
    'res2net34_26w_4s': {'parameter': dict(nblock=[3, 4, 6, 3], base_width=26, scale=4, block=BasicBlock), 'etc': {}},
    'res2net50_26w_4s': {'parameter': dict(nblock=[3, 4, 6, 3], base_width=26, scale=4, block=BottleNeck), 'etc': {}},
    'res2net101_26w_4s': {'parameter': dict(nblock=[3, 4, 23, 3], base_width=26, scale=4, block=BottleNeck), 'etc': {}},
    'res2net152_26w_4s': {'parameter': dict(nblock=[3, 8, 36, 3], base_width=26, scale=4, block=BottleNeck), 'etc': {}},

    # Res2Next
    'res2next34_8c_4w_4s': {'parameter': dict(nblock=[3, 4, 6, 3], groups=8, base_width=4, scale=4, block=BasicBlock), 'etc': {}},
    'res2next50_8c_4w_4s': {'parameter': dict(nblock=[3, 4, 6, 3], groups=8, base_width=4, scale=4, block=BottleNeck), 'etc': {}},
    'res2next101_8c_4s_4s': {'parameter': dict(nblock=[3, 4, 23, 3], groups=8, base_width=4, scale=4, block=BottleNeck), 'etc': {}},
    'res2next152_8c_4s_4s': {'parameter': dict(nblock=[3, 8, 36, 3], groups=8, base_width=4, scale=4, block=BottleNeck), 'etc': {}},
}


@register_model
class Res2Net(nn.Module):
    def __init__(self,
                 nblock,
                 block = BottleNeck,
                 norm_layer: nn.Module = nn.BatchNorm2d,
                 channels=[64, 128, 256, 512],
                 strides=[1, 2, 2, 2],
                 scale=4,
                 groups=1,
                 base_width=64,
                 zero_init_last=True,
                 num_classes=1000,
                 in_channels=3,
                 drop_path_rate=0.0,) -> None:
        super(Res2Net, self).__init__()
        self.groups = groups
        self.num_classes = num_classes
        self.base_width = base_width
        self.scale = scale
        self.norm_layer = norm_layer
        self.in_channels = channels[0] * scale
        self.out_channels = channels[-1] * block.factor
        self.num_block = sum(nblock)
        self.cur_block = 0
        self.drop_path_rate = drop_path_rate

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
                                scale=self.scale, base_width=self.base_width, drop_path_rate=self.get_drop_path_rate()))
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



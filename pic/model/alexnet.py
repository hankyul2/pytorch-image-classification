from torch import nn


class ConvActPool(nn.Sequential):
    def __init__(self, in_ch=1, out_ch=96, k_size=11, stride=4, pad=0, pool=True, pool_k_size=3, pool_stride=2):
        super().__init__(nn.Conv2d(in_ch, out_ch, k_size, stride, pad), nn.ReLU(inplace=True),
                       nn.MaxPool2d(pool_k_size, pool_stride) if pool else nn.Identity())

        
class LinearActDropout(nn.Sequential):
    def __init__(self, in_ch=9126, out_ch=4096, dropout=0.1):
        super().__init__(nn.Linear(in_ch, out_ch), nn.ReLU(inplace=True), nn.Dropout(p=dropout))


class AlexNet(nn.Module):
    def __init__(self, ch=3, dropout=0.5, nclass=10):
        super(AlexNet, self).__init__()
        self.convolution_layers = nn.ModuleList([
            #          in  out ks st pad pool
            ConvActPool(ch, 96, 11, 4, 0, True),
            ConvActPool(96, 256, 5, 1, 2, True),
            ConvActPool(256, 384, 3, 1, 1, False),
            ConvActPool(384, 384, 3, 1, 1, False),
            ConvActPool(384, 256, 3, 1, 1, True),
        ])
        self.flatten = nn.Flatten()
        self.fully_connected_layers = nn.Sequential(
            #                in       out   dropout
            LinearActDropout(256*6*6, 4096, dropout),
            LinearActDropout(4096, 4096, dropout),
        )
        self.classifier = nn.Linear(4096, nclass)

    def forward(self, x):
        for layer in self.convolution_layers:
            x = layer(x)
        x = self.flatten(x)
        x = self.fully_connected_layers(x)
        x = self.classifier(x)
        return x
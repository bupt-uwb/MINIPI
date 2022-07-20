from .UNet_blocks import *


class UNet_LA(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False, light_attention=True):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.ligtht_attention = light_attention

        self.inc = DoubleConv(n_channels, 16)
        self.down0 = Down(16, 32)
        self.down1 = Down(32, 64)
        self.up1 = Up(32, 16, bilinear, light_attention)
        self.up0 = Up(64, 32, bilinear, light_attention)
        self.outc = OutConv(16, n_classes)
        self.softmax = nn.Softmax(dim=3)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down0(x1)
        x3 = self.down1(x2)
        x = self.up0(x3, x2)
        x = self.up1(x, x1)
        output = self.outc(x)
        return self.softmax(output)


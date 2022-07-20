import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.double_conv(x)
        return x


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True, light_attention=False):
        super().__init__()
        self.light_attention = light_attention
        self.bilinear = bilinear
        if self.light_attention:
            self.Attention_Module = AttentionModule(out_channels)
        # if bilinear, use the normal convolutions to reduce the number of channels
        if self.bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.reduce_ch = nn.Conv2d(in_channels, in_channels // 2, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        if self.bilinear:
            x1 = self.reduce_ch(x1)
        x_cat = torch.cat([x2, x1], dim=1)
        if self.light_attention:
            x = self.Attention_Module(x1, x2, x_cat)
            x = self.conv(x)
        else:
            x = self.conv(x_cat)
        return x


class AttentionModule(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.Get_KQ = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels//4, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(in_channels//4),
            nn.ReLU(inplace=True)
        )
        self.Get_V = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=(1, 1), stride=(1, 1),
                      bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        self.reduce_channels = nn.Sequential(
            nn.Conv2d(in_channels=in_channels*2, out_channels=in_channels, kernel_size=(1, 1), stride=(1, 1),
                      bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x_en, x_de, x_cat):
        #Get AttentionMap
        H = x_en.size()[2]
        W = x_en.size()[3]
        N = H*W
        K = self.Get_KQ(x_en)
        K = K.reshape(K.size()[0], K.size()[1], N)
        Q = self.Get_KQ(x_de)
        Q = Q.reshape(Q.size()[0], Q.size()[1], N)
        KQ = torch.bmm(K.transpose(1, 2), Q)
        KQ = KQ.reshape(KQ.size()[0], N*N)
        KQ = self.softmax(KQ)
        AttentionMap = KQ.reshape(KQ.size()[0], N, N)
        #Get V
        x = self.reduce_channels(x_cat)
        V = self.Get_V(x)
        V = V.reshape(V.size()[0], V.size()[1], N)
        FeatureMap = torch.bmm(V, AttentionMap.transpose(1, 2)).reshape(V.size()[0], V.size()[1], H, W)
        return torch.cat([x, FeatureMap], dim=1)

class OutConv(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=(20, 1), stride=(1, 1))
        self.FC = nn.Linear(in_features=5, out_features=num_classes)

    def forward(self, x):
        x = self.conv(x)
        x = self.FC(x)
        return x
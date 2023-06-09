import torch
import torch.nn as nn
import torch.nn.functional as F
from .vgg import VGG


# VGG16, EORSSD的权重53, ORSSD的权重51, ORSI-4199的权重52
class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Reduction(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Reduction, self).__init__()
        self.reduce = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, 3, padding=1),
            BasicConv2d(out_channel, out_channel, 3, padding=1)
        )

    def forward(self, x):
        return self.reduce(x)


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv2d(in_planes // 16, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x1 = torch.cat([avg_out, max_out], dim=1)
        x2 = self.conv1(x1)
        return self.sigmoid(x2)


class MAI(nn.Module):
    def __init__(self, channel):
        super(MAI, self).__init__()
        self.ms = Multi_scale(channel, channel)
        self.ca = ChannelAttention(channel)
        self.sa = SpatialAttention()

        self.edg_conv = nn.Sequential(
            BasicConv2d(channel, channel, 3, padding=1),
            BasicConv2d(channel, channel, 3, padding=1)
        )
        self.sal_conv = nn.Sequential(
            BasicConv2d(2 * channel, channel, 3, padding=1),
            BasicConv2d(channel, channel, 3, padding=1)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, sal, edg):
        sal_r = self.ms(sal)
        sal_c = self.ca(sal_r) * sal_r
        sal_A = self.sa(sal_c) * sal_c
        edg_s = self.sigmoid(edg) * edg
        edg_o = self.edg_conv(edg_s * sal_A)

        sal_o = self.sal_conv(torch.cat((sal_A, edg_s), 1))

        return (sal + sal_o), (edg + edg_o)

class Multi_scale(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Multi_scale, self).__init__()
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.branch4 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 9), padding=(0, 4)),
            BasicConv2d(out_channel, out_channel, kernel_size=(9, 1), padding=(4, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=9, dilation=9)
        )
        self.branch5 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 11), padding=(0, 5)),
            BasicConv2d(out_channel, out_channel, kernel_size=(11, 1), padding=(5, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=11, dilation=11)
        )

        self.conv_cat = BasicConv2d(6 * out_channel, out_channel, 3, padding=1)


    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x + x0)
        x2 = self.branch2(x + x1)
        x3 = self.branch3(x + x2)
        x4 = self.branch4(x + x3)
        x5 = self.branch5(x + x4)
        x_cat = torch.cat((x0, x1, x2, x3, x4, x5), 1)
        x_cat = self.conv_cat(x_cat)
        return x_cat


class SF(nn.Module):
    def __init__(self, channel):
        super(SF, self).__init__()
        self.conv1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv2 = BasicConv2d(channel, channel, 3, padding=1)
        self.S_conv = nn.Sequential(
            BasicConv2d(2 * channel, channel, 3, padding=1),
            BasicConv2d(channel, channel, 1)
        )

    def forward(self, fl, fh, fs):
        fsl = F.interpolate(fs, size=fl.size()[2:], mode='bilinear')

        fh = self.conv1(fsl * fh + fh)
        fl = self.conv2(fsl * fl + fl)
        out = self.S_conv(torch.cat((fh, fl), 1))
        return out


class SEINet(nn.Module):
    def __init__(self, channel=64):
        super(SEINet, self).__init__()
        # Backbone model
        self.vgg = VGG('rgb')

        self.reduce_sal1 = Reduction(128, channel)
        self.reduce_sal2 = Reduction(256, channel)
        self.reduce_sal3 = Reduction(512, channel)
        self.reduce_sal4 = Reduction(512, channel)

        self.reduce_edg1 = Reduction(128, channel)
        self.reduce_edg2 = Reduction(256, channel)
        self.reduce_edg3 = Reduction(512, channel)
        self.reduce_edg4 = Reduction(512, channel)

        self.S1 = nn.Sequential(
            BasicConv2d(channel, channel, 3, padding=1),
            nn.Conv2d(channel, 1, 1)
        )
        self.S2 = nn.Sequential(
            BasicConv2d(channel, channel, 3, padding=1),
            nn.Conv2d(channel, 1, 1)
        )
        self.S3 = nn.Sequential(
            BasicConv2d(channel, channel, 3, padding=1),
            nn.Conv2d(channel, 1, 1)
        )
        self.S4 = nn.Sequential(
            BasicConv2d(channel, channel, 3, padding=1),
            nn.Conv2d(channel, 1, 1)
        )
        self.S5 = nn.Sequential(
            BasicConv2d(channel, channel, 3, padding=1),
            nn.Conv2d(channel, 1, 1)
        )
        self.S6 = nn.Sequential(
            BasicConv2d(channel, channel, 3, padding=1),
            nn.Conv2d(channel, 1, 1)
        )
        self.S7 = nn.Sequential(
            BasicConv2d(channel, channel, 3, padding=1),
            nn.Conv2d(channel, 1, 1)
        )
        self.S8 = nn.Sequential(
            BasicConv2d(channel, channel, 3, padding=1),
            nn.Conv2d(channel, 1, 1)
        )
        self.S_conv1 = nn.Sequential(
            BasicConv2d(2 * channel, channel, 3, padding=1),
            BasicConv2d(channel, channel, 1)
        )
        self.S_conv2 = nn.Sequential(
            BasicConv2d(2 * channel, channel, 3, padding=1),
            BasicConv2d(channel, channel, 1)
        )
        self.S_conv3 = nn.Sequential(
            BasicConv2d(2 * channel, channel, 3, padding=1),
            BasicConv2d(channel, channel, 1)
        )
        self.S_conv4 = nn.Sequential(
            BasicConv2d(2 * channel, channel, 3, padding=1),
            BasicConv2d(channel, channel, 1)
        )

        self.sigmoid = nn.Sigmoid()

        self.mai4 = MAI(channel)
        self.mai3 = MAI(channel)
        self.mai2 = MAI(channel)
        self.mai1 = MAI(channel)

        self.sf1 = SF(channel)
        self.sf2 = SF(channel)

    def forward(self, x):
        size = x.size()[2:]
        x1 = self.vgg.conv1(x)
        x2 = self.vgg.conv2(x1)
        x3 = self.vgg.conv3(x2)
        x4 = self.vgg.conv4(x3)
        x5 = self.vgg.conv5(x4)

        x_sal1 = self.reduce_sal1(x2)
        x_sal2 = self.reduce_sal2(x3)
        x_sal3 = self.reduce_sal3(x4)
        x_sal4 = self.reduce_sal4(x5)

        x_edg1 = self.reduce_edg1(x2)
        x_edg2 = self.reduce_edg2(x3)
        x_edg3 = self.reduce_edg3(x4)
        x_edg4 = self.reduce_edg4(x5)

        sal4, edg4 = self.mai4(x_sal4, x_edg4)

        sal4_3 = F.interpolate(sal4, size=x_sal3.size()[2:], mode='bilinear')
        edg4_3 = F.interpolate(edg4, size=x_sal3.size()[2:], mode='bilinear')

        x_sal3 = self.S_conv1(torch.cat((sal4_3, x_sal3), 1))
        x_edg3 = self.S_conv2(torch.cat((edg4_3, x_edg3), 1))

        sal3, edg3 = self.mai3(x_sal3, x_edg3)
        sal3_2 = F.interpolate(sal3, size=x_sal2.size()[2:], mode='bilinear')
        edg3_2 = F.interpolate(edg3, size=x_sal2.size()[2:], mode='bilinear')

        x_sal2 = self.sf1(x_sal2, sal3_2, sal4)
        x_edg2 = self.S_conv3(torch.cat((edg3_2, x_edg2), 1))

        sal2, edg2 = self.mai2(x_sal2, x_edg2)
        sal2_1 = F.interpolate(sal2, size=x_sal1.size()[2:], mode='bilinear')
        edg2_1 = F.interpolate(edg2, size=x_sal1.size()[2:], mode='bilinear')

        x_sal1 = self.sf2(x_sal1, sal2_1, sal4)
        x_edg1 = self.S_conv4(torch.cat((edg2_1, x_edg1), 1))

        sal1, edg1 = self.mai1(x_sal1, x_edg1)

        sal_out = self.S1(sal1)
        edg_out = self.S2(edg1)
        sal2 = self.S3(sal2)
        edg2 = self.S4(edg2)
        sal3 = self.S5(sal3)
        edg3 = self.S6(edg3)
        sal4 = self.S7(sal4)
        edg4 = self.S8(edg4)

        sal_out = F.interpolate(sal_out, size=size, mode='bilinear', align_corners=True)
        edg_out = F.interpolate(edg_out, size=size, mode='bilinear', align_corners=True)
        sal2 = F.interpolate(sal2, size=size, mode='bilinear', align_corners=True)
        edg2 = F.interpolate(edg2, size=size, mode='bilinear', align_corners=True)
        sal3 = F.interpolate(sal3, size=size, mode='bilinear', align_corners=True)
        edg3 = F.interpolate(edg3, size=size, mode='bilinear', align_corners=True)
        sal4 = F.interpolate(sal4, size=size, mode='bilinear', align_corners=True)
        edg4 = F.interpolate(edg4, size=size, mode='bilinear', align_corners=True)

        return sal_out, self.sigmoid(sal_out), edg_out, sal2, self.sigmoid(sal2), edg2,  sal3, self.sigmoid(sal3), edg3, sal4, self.sigmoid(sal4), edg4

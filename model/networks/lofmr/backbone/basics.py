import torch.nn as nn
import torch.nn.functional as F


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution without padding"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

'''class InitConv(nn.Module):
    def __init__(self, in_channels=4, out_channels=16, dropout=0.2):
        super(InitConv, self).__init__()

        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3,stride=2, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        y = self.relu(self.bn1(self.conv(x)))
        return y'''

class ConvBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1, dropout=0.2):
        super().__init__()
        self.conv = conv3x3(in_planes, planes, stride)
        self.bn = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = dropout

    def forward(self, x):
        x = self.conv(x)
        if self.dropout>0:
            x = F.dropout3d(x)
        return self.relu(self.bn(x))
class ResBasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1, dropout=0.2):
        super().__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.conv2 = conv3x3(planes, planes)
        self.bn1 = nn.BatchNorm3d(planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = dropout

        if stride == 1:
            self.downsample = None
        else:
            self.downsample = nn.Sequential(
                conv1x1(in_planes, planes, stride=stride),
                nn.BatchNorm3d(planes)
            )

    def forward(self, x):
        y = x
        if self.dropout>0:
            y = F.dropout3d(self.conv1(y), self.dropout)
        else:
            y =self.conv1(y)

        y = self.relu(self.bn1(y))
        y = self.bn2(self.conv2(y))

        if self.downsample is not None:
            x = self.downsample(x)
        return self.relu(x+y)


class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels,type='conv'):
        super(Downsample, self).__init__()
        if type is "conv":
            self.pool_op = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        else:
            self.pool_op = nn.MaxPool3d(kernel_size=[2,2,2])
    def forward(self, x):
        y = self.pool_op(x)
        return y

class UpInterpoate(nn.Module):
    def __init__(self, scale_factor=None,mode='trilinear', align_corners=False):
        super(UpInterpoate, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners
    def forward(self,x):
        return F.interpolate(x,scale_factor=self.scale_factor,mode=self.mode,align_corners=self.align_corners)

class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels,type='interpoate'):
        super(Upsample, self).__init__()
        if type is "conv":
            self.up_op = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=[2,2,2], stride=[2,2,2], padding=1)
        else:
            self.up_op = UpInterpoate(scale_factor=2., mode='trilinear', align_corners=True)

    def forward(self, x):
        y = self.up_op(x)
        return y
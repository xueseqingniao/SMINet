 # coding=utf-8
from turtle import forward
import torch
import torch.nn as nn
from torch.nn import Parameter
from torch.nn.modules.conv import Conv2d
#from model.resattention import res_cbam
import torchvision.models as models
#from model.res2fg import res2net
import torch.nn.functional as F
import math
# Low level
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)#dim=1是因为传入数据是多加了一维unsqueeze(0)
        x=max_out
        x = self.conv1(x)
        return self.sigmoid(x)


#mid level
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
        return x



class DCA(nn.Module):
    def __init__(self,in_channel):
        super(DCA,self).__init__()
        self.ca1 = ChannelAttention(in_channel)
        self.ca2 = ChannelAttention(in_channel)
        self.sa1 = SpatialAttention()
        self.sa2 = SpatialAttention()
    def forward(self,x):
        x0,x1,x2,x3 = torch.split(x,x.size()[1]//4,dim=1)
        v0 = self.ca1(x0)
        z0 = x0*v0+x0
        
        v2 = self.ca2(x2)
        z2 = x2*v2+x2

        v1 = self.sa1(x1)
        z1 = x1*v1+x1

        v3 = self.sa2(x3)
        z3 = x3*v3+x3

        z = torch.cat((z0,z1,z2,z3),1)
        return z
class SFE(nn.Module):
    def __init__(self,in_channel):
        super(SFE,self).__init__()
        self.weight = Parameter(torch.FloatTensor(in_channel,in_channel),requires_grad=True)
        self.weight2 = Parameter(torch.FloatTensor(in_channel,in_channel),requires_grad=True)
        self.reset_para()
        self.ca = ChannelAttention(2*in_channel)
        self.conv = nn.Conv2d(2*in_channel,in_channel,kernel_size=1)
    def reset_para(self):
        stdv=1./math.sqrt(self.weight.size(1))
        stdv2=1./math.sqrt(self.weight2.size(1))
        self.weight.data.uniform_(-stdv,stdv)
        self.weight2.data.uniform_(-stdv2,stdv2)
    def forward(self,x1,x2):
        batch_size = x2.size(0)
        channel = x2.size(1)
        #print(channel)
        g_x = x2.view(batch_size, channel, -1)#[bs, c, w*h]
        g_x = g_x.permute(0, 2, 1)#[b,wh,c]
        theta_x = x1.view(batch_size, channel, -1)
        theta_x = theta_x.permute(0, 2, 1)#[b,wh,c]

        phi_x = x2.view(batch_size, channel, -1)#[bs, c, w*h]
        
        f = torch.matmul(theta_x, phi_x)#[b,wh,wh]

        adj = F.softmax(f, dim=-1)
        support = torch.matmul(g_x,self.weight)#[b,wh,c]
        y = torch.matmul(adj,support)#[b,wh,c]
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, channel, *x2.size()[2:])
        
        a_x = x1.view(batch_size, channel, -1)
        b_x = x2.view(batch_size, channel, -1)
        b_x = b_x.permute(0,2,1)
        c_x = torch.matmul(a_x,b_x)#[c,c]
        support2 = torch.matmul(c_x,self.weight2)
        y2 = torch.matmul(support2,a_x).contiguous()
        y2 = y2.view(batch_size, channel, *x2.size()[2:])
        
        z = torch.cat((y,y2),1)
        v = self.ca(z)
        out = z*v+z
        out = self.conv(out)
        return out
class CBR(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(CBR,self).__init__()
        self.conv = nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1)
        self.Ba = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
    def forward(self,x):
        out = self.relu(self.Ba(self.conv(x)))

        return out
class SMINet(nn.Module):#输入三通道
    def __init__(self, in_channels):
        super(SMINet, self).__init__()
        resnet = models.resnet18(pretrained=True)
        channel = 64
        #resnet2 = models.resnet18(pretrained=True)
        #self.weight=nn.Parameter(torch.FloatTensor(1))
        #
        #reanet = res2net()
        #res2n
        # ************************* Encoder ***************************
        # input conv3*3,64
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)#计算得到112.5 但取112 向下取整
        # Extract Features
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4
        
        # ************************* Decoder ***************************
        
        # ************************* Feature Map Upsample ***************************
        self.downsample = nn.Upsample(scale_factor=0.5, mode='bilinear')
        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.upsample2 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.upsample3 = nn.Upsample(scale_factor=8, mode='bilinear')
        self.upsample4 = nn.Upsample(scale_factor=16, mode='bilinear')
        
        self.conv_1x1_0 = nn.Conv2d(64,channel,kernel_size=3,padding=1)
        self.conv_1x1_1 = nn.Conv2d(64,channel,kernel_size=3,padding=1)
        self.conv_1x1_2 = nn.Conv2d(128,channel,kernel_size=3,padding=1)
        self.conv_1x1_3 = nn.Conv2d(256,channel,kernel_size=3,padding=1)
        self.conv_1x1_4 = nn.Conv2d(512,channel,kernel_size=3,padding=1)
        
        self.cbr0 = CBR(896,128)
        self.cbr1 = CBR(576,128)
        self.cbr2 = CBR(320,128)
        self.cbr3 = CBR(128,128)
        self.cbr4 = CBR(64,128)

        self.sfe = SFE(128)
        self.dca0 = DCA(32)
        self.dca1 = DCA(32)

        self.conv_sal2 = CBR(256,128)
        self.conv_sal1 = CBR(256,128)
        self.conv_sal0 = CBR(256,128)
        self.conv_1x1_N = nn.Conv2d(128,1,kernel_size=3,padding=1)
        self.conv_1x1_S0 = nn.Conv2d(128,1,kernel_size=3,padding=1)
        self.conv_1x1_S1 = nn.Conv2d(128,1,kernel_size=3,padding=1)
        self.conv_1x1_S3 = nn.Conv2d(128,1,kernel_size=3,padding=1)
    def forward(self, x):
        # ************************* Encoder ***************************
        #print(x.size())
        
        # input
        tx = self.conv1(x)
        tx = self.bn1(tx)
        f0 = self.relu(tx)
        tx = self.maxpool(f0)
        # Extract Features
        f1 = self.encoder1(tx)
        f2 = self.encoder2(f1)
        f3 = self.encoder3(f2)
        f4 = self.encoder4(f3)
        
        f0 = self.conv_1x1_0(f0)
        f1 = self.conv_1x1_1(f1)
        f2 = self.conv_1x1_2(f2)
        f3 = self.conv_1x1_3(f3)
        f4 = self.conv_1x1_4(f4)
        #128
        f0_1 = torch.cat((f0,self.upsample1(f1)),1) 
        f1_1 = torch.cat((f1,self.upsample1(f2)),1)
        f2_1 = torch.cat((f2,self.upsample1(f3)),1)
        f3_1 = torch.cat((f3,self.upsample1(f4)),1)
        #192
        f0_2 = torch.cat((f0,self.upsample1(f1),self.upsample2(f2)),1)
        f1_2 = torch.cat((f1,self.upsample1(f2),self.upsample2(f3)),1)
        f2_2 = torch.cat((f2,self.upsample1(f3),self.upsample2(f4)),1)
        #256
        f0_3 = torch.cat((f0,self.upsample1(f1),self.upsample2(f2),self.upsample3(f3)),1)
        f1_3 = torch.cat((f1,self.upsample1(f2),self.upsample2(f3),self.upsample3(f4)),1)
        #320
        f0_4 = torch.cat((f0,self.upsample1(f1),self.upsample2(f2),self.upsample3(f3),self.upsample4(f4)),1)

        F0 = torch.cat((f0_1,f0_2,f0_3,f0_4),1)#896
        F1 = torch.cat((f1_1,f1_2,f1_3),1)#576
        F2 = torch.cat((f2_1,f2_2),1)#320
        F3 = f3_1#128
        F4 = f4#64
        
        F0 = self.cbr0(F0)
        F1 = self.cbr1(F1)
        F2 = self.cbr2(F2)
        F3 = self.cbr3(F3)
        F4 = self.cbr4(F4)
        
        N = self.sfe(self.upsample1(F3),F2)
        S0 = self.dca0(F0)
        S1 = self.dca1(F1)

        sal2 = self.conv_sal2(torch.cat((N,self.upsample2(F4)),1))
        sal1 = self.conv_sal1(torch.cat((S1,self.upsample1(sal2)),1))
        sal0 = self.conv_sal0(torch.cat((S0,self.upsample1(sal1)),1))
        
        s3 = F.sigmoid(self.conv_1x1_S3(F4))
        s2 = F.sigmoid(self.conv_1x1_N(sal2))
        s1 = F.sigmoid(self.conv_1x1_S1(sal1))
        s0 = F.sigmoid(self.conv_1x1_S0(sal0))
        return s0,self.upsample1(s1),self.upsample2(s2),self.upsample4(s3)



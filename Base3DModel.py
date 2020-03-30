import torch
import torch.nn as nn

class Base3DModel(nn.Module):
    def __init__(self,num_seg_classes,f=8):
        super(Base3DModel,self).__init__()
        
        self.conv = nn.Sequential()
        #in_channels(int) – 输入信号的通道，就是输入中每帧图像的通道数
        #out_channels(int) – 卷积产生的通道，就是输出中每帧图像的通道数
        #kernel_size(int or tuple) - 过滤器的尺寸，假设为(a,b,c)，表示的是过滤器每次处理 a 帧图像，该图像的大小是b x c。
        #stride(int or tuple, optional) - 卷积步长，形状是三维的，假设为(x,y,z)，表示的是三维上的步长是x，在行方向上步长是y，在列方向上步长是z。
        #padding(int or tuple, optional) - 输入的每一条边补充0的层数，形状是三维的，假设是(l,m,n)，表示的是在输入的三维方向前后分别padding l 个全零二维矩阵，在输入的行方向上下分别padding m 个全零行向量，在输入的列方向左右分别padding n 个全零列向量。
        #dilation(int or tuple, optional) – 卷积核元素之间的间距，这个看看空洞卷积就okay了
        self.conv.add_module('conv1', nn.Conv3d(in_channels=1, out_channels=4 * f, kernel_size=1, stride=1, padding=0, dilation=1))
        self.conv.add_module('conv2', nn.InstanceNorm3d(num_features=4 * f))
        self.conv.add_module('conv3', nn.ReLU(inplace=True))
        self.conv.add_module('conv4', nn.MaxPool3d(kernel_size=3, stride=2))

        self.conv.add_module('conv5', nn.Conv3d(in_channels=4 * f, out_channels=32 * f, kernel_size=2, stride=1, padding=0, dilation=2))
        self.conv.add_module('conv6', nn.InstanceNorm3d(num_features=32 * f))
        self.conv.add_module('conv7', nn.ReLU(inplace=True))
        self.conv.add_module('conv8', nn.MaxPool3d(kernel_size=3, stride=2))

        self.conv.add_module('conv9', nn.Conv3d(in_channels=32 * f, out_channels=64 * f, kernel_size=2, stride=1, padding=2, dilation=2))
        self.conv.add_module('conv10', nn.InstanceNorm3d(num_features=64 * f))
        self.conv.add_module('conv11', nn.ReLU(inplace=True))
        self.conv.add_module('conv12', nn.MaxPool3d(kernel_size=3, stride=2))

        self.conv.add_module('conv13', nn.Conv3d(in_channels=64 * f, out_channels=64 * f, kernel_size=2, stride=1, padding=1, dilation=2))
        self.conv.add_module('conv14', nn.InstanceNorm3d(num_features=64 * f))
        self.conv.add_module('conv15', nn.ReLU(inplace=True))
        self.conv.add_module('conv16', nn.MaxPool3d(kernel_size=5, stride=2))

        self.fc = nn.Sequential()
        self.fc.add_module('fc1', nn.Linear(64 * f * 3 * 4 * 3, 1024))
        self.fc.add_module('dp1', nn.Dropout(0.3))
        self.fc.add_module('fc2', nn.Linear(1024, 3))
        
    def forward(self,x):
        z = self.conv(x)
        z = self.fc(z.view(x.shape[0],-1))
        
        return z
import torch
import torch.nn as nn

class SRCNN(nn.Module):
    def __init__(self):
        super(SRCNN, self).__init__()
        # 第一个卷积层：将256通道减少到64通道
        self.conv1 = nn.Conv2d(256, 64, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        # 第二个卷积层：将64通道减少到32通道
        self.conv2 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        # 上采样层：将空间尺寸从32x48增加到192x288（6倍上采样）
        self.upsample = nn.ConvTranspose2d(32, 32, kernel_size=6, stride=6, padding=0)
        # 第三个卷积层：将32通道减少到1通道，生成最终输出
        self.conv3 = nn.Conv2d(32, 1, kernel_size=3, padding=1)

    def forward(self, x):
        # 前向传播
        x = self.relu1(self.conv1(x))  # 特征提取
        x = self.relu2(self.conv2(x))  # 非线性映射
        x = self.upsample(x)           # 上采样
        x = self.conv3(x)              # 输出层
        return x

# 测试模型
model = SRCNN()
img = torch.randn(1, 256, 32, 48)
out = model(img)
print(out.shape)  # 输出：torch.Size([1, 1, 192, 288])
#https://github.com/yjn870/SRCNN-pytorch/tree/master
'''
数据相关参数
训练数据文件路径（--train-file）：必须指定，用于加载训练数据集，格式为 HDF5。
评估数据文件路径（--eval-file）：必须指定，用于加载评估数据集，格式为 HDF5。
训练过程参数
学习率（--lr）：默认值为 1e-4，用于优化器调整模型参数更新步长。
批量大小（--batch-size）：默认值为 16，即每次训练时输入模型的样本数量。
训练轮数（--num-epochs）：默认值为 400，模型对整个训练数据集进行训练的次数。
数据加载线程数（--num-workers）：默认值为 8，用于多线程加载数据，提高数据加载效率。
模型相关参数
放大比例（--scale）：默认值为 3，用于指定图像超分辨率的放大倍数，可选值有 2、3、4 。
其他参数
输出目录（--outputs-dir）：必须指定，用于保存训练过程中产生的模型权重文件，会根据放大比例在该目录下创建子目录。
随机种子（--seed）：默认值为 123，用于设置随机数生成器的种子，以确保实验的可重复性
'''
import torch  
import torch.nn as nn  
import torch.nn.functional as F  

class ResidualBlock(nn.Module):  
    def __init__(self, in_channels, out_channels, stride=1):  
        super(ResidualBlock, self).__init__()  
        
        # Convolutional layers  
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)  
        self.bn1 = nn.BatchNorm2d(out_channels, momentum=0.5)  
        
        # Parametric ReLU (PReLU)  
        self.prelu1 = nn.PReLU(num_parameters=1, init=0)  
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)  
        self.bn2 = nn.BatchNorm2d(out_channels, momentum=0.5)  
    
    def forward(self, x):  
        residual = x  
        
        out = self.conv1(x)  
        out = self.bn1(out)  
        out = self.prelu1(out)  
        
        out = self.conv2(out)  
        out = self.bn2(out)  
        
        out += residual  
        return out  

class Generator(nn.Module):  
    def __init__(self):  
        super(Generator, self).__init__()  
        noise_shape=(256, 32, 48)
        # Initial convolution  
        self.init_conv = nn.Conv2d(noise_shape[0], 64, kernel_size=3, stride=1, padding=1)  
        self.init_prelu = nn.PReLU(num_parameters=1, init=0)  
        
        # Residual blocks  
        self.res_blocks = nn.ModuleList([  
            ResidualBlock(64, 64) for _ in range(16)  
        ])  
        
        # Additional convolution after residual blocks  
        self.post_res_conv = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)  
        self.post_res_bn = nn.BatchNorm2d(64, momentum=0.5)  
        
        # Upsampling blocks  
        self.upsample_blocks = nn.ModuleList([  
            nn.Sequential(  
                nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  
                # nn.Upsample(scale_factor=2, mode='nearest'),  
                nn.PReLU(num_parameters=1, init=0),  
                
                nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),  
                nn.Upsample(scale_factor=3, mode='nearest'),  
                nn.PReLU(num_parameters=1, init=0),  
                
                nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),  
                nn.Upsample(scale_factor=2, mode='nearest'),  
                nn.PReLU(num_parameters=1, init=0)  
            )  
        ])  
        
        # Final convolution  
        self.final_conv = nn.Conv2d(128, 1, kernel_size=9, stride=1, padding=4)  
    
    def forward(self, x):  
        # Initial convolution  
        x = self.init_conv(x)  
        x = self.init_prelu(x)  
        
        # Store for residual connection  
        residual = x  
        
        # Residual blocks  
        for res_block in self.res_blocks:  
            x = res_block(x)  
        
        # Post-residual convolution  
        x = self.post_res_conv(x)  
        x = self.post_res_bn(x)  
        x = x + residual  
        
        # Upsampling  
        x = self.upsample_blocks[0](x)  
        
        # Final convolution  
        x = self.final_conv(x)  
        
        return x  

# Test the generator  
# def test_generator():  
#     # Test with the provided input shape  
#     img = torch.randn(1, 256, 32, 48)  
    
#     # Initialize generator  
#     generator = Generator(noise_shape=(256, 32, 48))  
    
#     # Generate output  
#     output = generator(img)  
    
#     print("Input shape:", img.shape)  
#     print("Output shape:", output.shape)  

# Run the test  
# test_generator()
#https://github.com/honghong2023/SRDRN/blob/master/code/train2.py

'''
#SRDRN: a DL approach, namely, Super Resolution Deep Residual Network (SRDRN). ###References: Wang, F., Tian, D., & Carroll, M. (2023). Customized deep learning for precipitation bias correction and downscaling. Geoscientific Model Development, 16(2), 535-556.
训练轮数（epochs）：在调用train函数时指定为 150，即模型对整个训练数据集进行训练的次数。
批量大小（batch_size）：在调用train函数时指定为 2，即每次训练时输入模型的样本数量。
学习率（lr）：在模型编译时，使用Adam优化器，学习率设置为 0.0001。
优化器参数（beta_1）：在模型编译时，Adam优化器的beta_1参数设置为 0.9。
损失函数：使用mae（平均绝对误差）作为损失函数。
评估指标：在模型编译时，指定评估指标为['mae', 'mse']，即平均绝对误差和均方误差。
'''
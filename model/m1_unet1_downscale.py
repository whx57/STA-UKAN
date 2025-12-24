import torch
import torch.nn as nn
import torch.nn.functional as F
#验证后上采样的效果
class UNetDownscale(nn.Module):
    def __init__(self, in_channels=256, out_channels=1, initial_feature_maps=256):
        super(UNetDownscale, self).__init__()
        
  
        # 编码器（下采样路径）
        self.enc1 = self.conv_block(in_channels, initial_feature_maps)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 32x48 -> 16x24
        
        self.enc2 = self.conv_block(initial_feature_maps, initial_feature_maps*2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 16x24 -> 8x12
        

        
        # 瓶颈层
        self.bottleneck = self.conv_block(initial_feature_maps*2, initial_feature_maps*4)
        
        # 解码器（上采样路径）

        
        self.up2 = nn.ConvTranspose2d(initial_feature_maps*4, initial_feature_maps*2, kernel_size=2, stride=2)  # 8x12 -> 16x24
        self.dec2 = self.conv_block(initial_feature_maps*4, initial_feature_maps*2)
        
        self.up1 = nn.ConvTranspose2d(initial_feature_maps*2, initial_feature_maps, kernel_size=2, stride=2)  # 16x24 -> 32x48
        self.dec1 = self.conv_block(initial_feature_maps*2, initial_feature_maps)
        
        # 上采样到最终的目标空间尺寸
        self.final_conv = nn.Conv2d(initial_feature_maps, out_channels, kernel_size=1)
        # self.up_final = nn.Upsample(scale_factor=6, mode='bilinear', align_corners=True)  # 32x48 -> 192x288
        
        # 上采样到最终的目标空间尺寸
        self.up_final = nn.Upsample(scale_factor=6, mode='bilinear', align_corners=True)  # 32x48 -> 192x288
        
        # 最终卷积层，将通道数降至输出要求
        self.final_conv = nn.Conv2d(initial_feature_maps, out_channels, kernel_size=1)
        
    def conv_block(self, in_channels, out_channels):
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        return block
        
    def forward(self, x):
        # 编码器路径
        # print('enc1 ')

        enc1 = self.enc1(x)       # [batch, initial_feature_maps, 32, 48]
        pool1 = self.pool1(enc1)  # [batch, initial_feature_maps, 16, 24]
        # print('enc2 ')
        enc2 = self.enc2(pool1)   # [batch, initial_feature_maps*2, 16, 24]
        pool2 = self.pool2(enc2)  # [batch, initial_feature_maps*2, 8, 12]
        # print('enc3 ')

        
        # 瓶颈层
        bottleneck = self.bottleneck(pool2)  # [batch, initial_feature_maps*8, 4, 6]
        # print('bottleneck ')
        # 解码器路径
        # print('up3 ')
    
        # print('up2 ')
        up2 = self.up2(bottleneck)                 
        up2 = torch.cat([up2, enc2], dim=1)  
        dec2 = self.dec2(up2)                
        # print('up1 ')
        up1 = self.up1(dec2)                 
        up1 = torch.cat([up1, enc1], dim=1)   
        # print('dec1 ')
        # print(up1.shape)
        dec1 = self.dec1(up1)   
        # print('up_final ')
        # 后上采样到目标空间尺寸
        up_final = self.up_final(dec1)         # [batch, initial_feature_maps, 192, 288]
        
        # 最终输出
        out = self.final_conv(up_final)        # [batch, 192, 192, 288]
        return out

if __name__ == "__main__":

    model1 = UNetDownscale(256, 1, 256)
    #打印模型参数量
    print(sum(p.numel() for p in model1.parameters()))
    x = torch.randn(1, 256, 32, 48)  
    out = model1(x)
    print(out.shape)  

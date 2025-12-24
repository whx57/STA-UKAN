import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import xarray as xr
import numpy as np
import torch
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from torch import nn
import sys
sys.path.append("/home/nxd/wx/wx/downscale_and_bias_correction")
from model.week5.kan import KAN, KANLinear
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from timm.models.layers import DropPath, trunc_normal_
from model.week5.kan import KANLinear

class SubPixelConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, upscale_factor):
        super(SubPixelConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels * (upscale_factor ** 2), kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
        self.drop = nn.Dropout(0.1)

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        return x

class KANLayer(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        grid_size = 5
        spline_order = 3
        scale_noise = 0.1
        scale_base = 1.0
        scale_spline = 1.0
        base_activation = torch.nn.SiLU
        grid_eps = 0.02
        grid_range = [-3, 3]

        self.fc1 = KANLinear(in_features, hidden_features, grid_size=grid_size, spline_order=spline_order,
                             scale_noise=scale_noise, scale_base=scale_base, scale_spline=scale_spline,
                             base_activation=base_activation, grid_eps=grid_eps, grid_range=grid_range)
        self.fc2 = KANLinear(hidden_features, out_features, grid_size=grid_size, spline_order=spline_order,
                             scale_noise=scale_noise, scale_base=scale_base, scale_spline=scale_spline,
                             base_activation=base_activation, grid_eps=grid_eps, grid_range=grid_range)
        self.fc3 = KANLinear(hidden_features, out_features, grid_size=grid_size, spline_order=spline_order,
                             scale_noise=scale_noise, scale_base=scale_base, scale_spline=scale_spline,
                             base_activation=base_activation, grid_eps=grid_eps, grid_range=grid_range)

        self.dwconv_1 = DW_bn_relu(hidden_features)
        self.dwconv_2 = DW_bn_relu(hidden_features)
        self.dwconv_3 = DW_bn_relu(hidden_features)

        self.drop = nn.Dropout(drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        B, N, C = x.shape

        x = self.fc1(x.reshape(B * N, C))
        x = x.reshape(B, N, C).contiguous()
        x = self.dwconv_1(x, H, W)
        x = self.fc2(x.reshape(B * N, C))
        x = x.reshape(B, N, C).contiguous()
        x = self.dwconv_2(x, H, W)
        x = self.fc3(x.reshape(B * N, C))
        x = x.reshape(B, N, C).contiguous()
        x = self.dwconv_3(x, H, W)

        return x

class KANBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=False, qk_scale=None, drop=0.0, attn_drop=0.0,
                 drop_path=0.0, act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)

        self.layer = KANLayer(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = x + self.drop_path(self.layer(self.norm2(x), H, W))
        return x

class DW_bn_relu(nn.Module):
    def __init__(self, dim=768):
        super(DW_bn_relu, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)
        self.bn = nn.BatchNorm2d(dim)
        self.relu = nn.ReLU()

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = x.flatten(2).transpose(1, 2)
        return x

class OverlapPatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=4, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = (192, 288)
        patch_size = (patch_size, patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W

class ResidualConvLayer(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ResidualConvLayer, self).__init__()
        self.conv3x3 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.conv5x5 = nn.Conv2d(in_ch, out_ch, kernel_size=5, padding=2)
        self.conv7x7 = nn.Conv2d(in_ch, out_ch, kernel_size=7, padding=3)

        self.bn = nn.BatchNorm2d(out_ch * 3)
        self.relu = nn.ReLU(inplace=True)

        self.conv1x1 = nn.Conv2d(out_ch * 3, out_ch, kernel_size=1)

        self.se = SEBlock(out_ch)

        if in_ch != out_ch:
            self.skip_conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)
            self.skip_bn = nn.BatchNorm2d(out_ch)
        else:
            self.skip_conv = None

    def forward(self, x):
        identity = x
        if self.skip_conv is not None:
            identity = self.skip_bn(self.skip_conv(x))

        out3x3 = self.conv3x3(x)
        out5x5 = self.conv5x5(x)
        out7x7 = self.conv7x7(x)

        out = torch.cat([out3x3, out5x5, out7x7], dim=1)

        out = self.bn(out)
        out = self.relu(out)

        out = self.conv1x1(out)

        out = self.se(out)

        out += identity
        out = self.relu(out)
        return out

class SEBlock(nn.Module):
    def __init__(self, in_ch, reduction=16):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Linear(in_ch, in_ch // reduction)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(in_ch // reduction, in_ch)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size, channels, _, _ = x.size()
        y = F.adaptive_avg_pool2d(x, 1).view(batch_size, channels)
        y = self.relu(self.fc1(y))
        y = self.sigmoid(self.fc2(y)).view(batch_size, channels, 1, 1)
        return x * y

class UKAN(nn.Module):
    def __init__(self, num_classes, input_channels=3, img_size=224, embed_dims=[256, 512, 1024], num_heads=[1, 2, 4, 8],
                 mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.0, attn_drop_rate=0.0,
                 drop_path_rate=0.0, norm_layer=nn.LayerNorm, depths=[1, 1, 1], sr_ratios=[8, 4, 2, 1], **kwargs):
        super().__init__()

        kan_input_dim = embed_dims[0]

        self.encoder1 = ResidualConvLayer(input_channels, kan_input_dim // 4)
        self.encoder2 = ResidualConvLayer(kan_input_dim // 4 + 1, kan_input_dim // 2)
        self.encoder3 = ResidualConvLayer(kan_input_dim // 2 + 1, kan_input_dim)

        self.norm3 = norm_layer(embed_dims[1])
        self.norm4 = norm_layer(embed_dims[2])

        self.dnorm3 = norm_layer(embed_dims[1])
        self.dnorm4 = norm_layer(embed_dims[0])

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.block1 = nn.ModuleList(
            [KANBlock(dim=embed_dims[1], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
                      drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[0], norm_layer=norm_layer,
                      sr_ratio=sr_ratios[0])]
        )

        self.block2 = nn.ModuleList(
            [KANBlock(dim=embed_dims[2], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
                      drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[1], norm_layer=norm_layer,
                      sr_ratio=sr_ratios[0])]
        )

        self.dblock1 = nn.ModuleList(
            [KANBlock(dim=embed_dims[1], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
                      drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[0], norm_layer=norm_layer,
                      sr_ratio=sr_ratios[0])]
        )

        self.dblock2 = nn.ModuleList(
            [KANBlock(dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
                      drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[1], norm_layer=norm_layer,
                      sr_ratio=sr_ratios[0])]
        )

        self.patch_embed3 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2, in_chans=embed_dims[0] + 1,
                                              embed_dim=embed_dims[1])
        self.patch_embed4 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2, in_chans=embed_dims[1] + 1,
                                              embed_dim=embed_dims[2])

        self.decoder1 = SubPixelConv2d(embed_dims[2] + 1, embed_dims[1], upscale_factor=2)
        self.decoder2 = SubPixelConv2d(embed_dims[1], embed_dims[0], upscale_factor=2)
        self.decoder3 = SubPixelConv2d(embed_dims[0], embed_dims[0] // 2, upscale_factor=2)
        self.decoder4 = SubPixelConv2d(embed_dims[0] // 2, embed_dims[0] // 4, upscale_factor=2)
        self.decoder5 = SubPixelConv2d(embed_dims[0] // 4, embed_dims[0] // 4, upscale_factor=2)

        self.final = nn.Conv2d(embed_dims[0] // 4, num_classes, kernel_size=1)
        self.up_final1 = Upsample(6, 256)
        self.map_path = '/home/nxd/wx/wx/downscale_and_bias_correction/data/final_map1.npy'
        self.load_map()

    def load_map(self):
        try:
            new_map = np.load(self.map_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"地图文件未找到: {self.map_path}")
        except Exception as e:
            raise ValueError(f"加载地图文件时发生错误: {e}")

        expected_shape = (192, 288)
        if new_map.shape != expected_shape:
            raise ValueError(f"预期的地图形状为 {expected_shape}，但得到 {new_map.shape}")

        if np.isnan(new_map).any() or np.isinf(new_map).any():
            raise ValueError("地图数据包含 NaN 或 Inf。请检查数据预处理步骤。")

        new_map_tensor = torch.tensor(new_map, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        self.register_buffer('new_map', new_map_tensor)

    def addmap(self, x, h, w, c, replace=True):
        B, C, H_x, W_x = x.shape
        if replace:
            if C <= c:
                raise ValueError(f"Expected at least {c + 1} channels, but got {C}")
            if H_x < h or W_x < w:
                raise ValueError(f"Input resolution ({H_x}, {W_x}) is smaller than target resolution ({h}, {w})")

            new_map_resized = F.interpolate(self.new_map, size=(h, w), mode='bilinear', align_corners=False)
            new_map_expanded = new_map_resized.expand(B, -1, -1, -1).to(x.device, x.dtype)
            x = torch.cat([x[:, :c, :, :], new_map_expanded], dim=1)
            return x
        else:
            new_map_resized = F.interpolate(self.new_map, size=(h, w), mode='bilinear', align_corners=False)
            new_map_expanded = new_map_resized.expand(B, -1, -1, -1).to(x.device, x.dtype)
            x = torch.cat([x, new_map_expanded], dim=1)
            return x

    def forward(self, x):
        x = self.up_final1(x)
        B, C, H, W = x.shape
        x = self.addmap(x, h=H, w=W, c=C - 1, replace=True)

        out = F.relu(F.max_pool2d(self.encoder1(x), 2, 2))
        t1 = out
        out = torch.cat([out, out[:, -1:, :, :]], dim=1)
        B, C_new, H, W = out.shape
        out = self.addmap(out, h=H, w=W, c=C_new - 1, replace=True)

        out = F.relu(F.max_pool2d(self.encoder2(out), 2, 2))
        t2 = out
        out = torch.cat([out, out[:, -1:, :, :]], dim=1)
        B, C_new, H, W = out.shape
        out = self.addmap(out, h=H, w=W, c=C_new - 1, replace=True)

        out = F.relu(F.max_pool2d(self.encoder3(out), 2, 2))
        t3 = out
        out = torch.cat([out, out[:, -1:, :, :]], dim=1)
        B, C_new, H, W = out.shape
        out = self.addmap(out, h=H, w=W, c=C_new - 1, replace=True)

        out, H, W = self.patch_embed3(out)
        for i, blk in enumerate(self.block1):
            out = blk(out, H, W)
        out = self.norm3(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        B, C_new, H, W = out.shape
        t4 = out
        out = torch.cat([out, out[:, -1:, :, :]], dim=1)
        B, C_new, H, W = out.shape
        out = self.addmap(out, h=H, w=W, c=C_new - 1, replace=True)

        out, H, W = self.patch_embed4(out)
        for i, blk in enumerate(self.block2):
            out = blk(out, H, W)
        out = self.norm4(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        B, C_new, H, W = out.shape
        out = torch.cat([out, out[:, -1:, :, :]], dim=1)
        B, C_new, H, W = out.shape
        out = self.addmap(out, h=H, w=W, c=C_new - 1, replace=True)

        out = F.relu(self.decoder1(out))
        out = torch.add(out, t4)
        B, C_new, H, W = out.shape
        out = out.flatten(2).transpose(1, 2)
        for i, blk in enumerate(self.dblock1):
            out = blk(out, H, W)

        out = self.dnorm3(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        out = F.relu(self.decoder2(out))
        out = torch.add(out, t3)
        B, C_new, H, W = out.shape
        out = out.flatten(2).transpose(1, 2)
        for i, blk in enumerate(self.dblock2):
            out = blk(out, H, W)

        out = self.dnorm4(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        out = F.relu(self.decoder3(out))
        out = torch.add(out, t2)

        out = F.relu(self.decoder4(out))
        out = torch.add(out, t1)

        out = F.relu(self.decoder5(out))

        return self.final(out)

class Upsample(nn.Sequential):
    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        elif scale == 6:
            m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(2))
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. Supported scales: 2^n, 3, and 6.')

        super(Upsample, self).__init__(*m)

if __name__ == "__main__":
    embed_dims = [128, 256, 512]
    model1 = UKAN(1, 256, 1, embed_dims)
    print(sum(p.numel() for p in model1.parameters()))
    x = torch.randn(1, 256, 32, 48)
    out = model1(x)
    print(out.shape)
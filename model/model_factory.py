# model_factory.py

# 导入所有模型类
from model.week5.m1_unet_downscale import UNetDownscale as UNet
from model.week5.m1_unet1_downscale import UNetDownscale as UNet1  # hou
from model.week5.m1_unet_downscale_transpose import UNetDownscale as UNet2
from model.week5.m1_unet1_downscale_transpose import UNetDownscale as UNet3  # hou
from model.week5.m2_c_unet_downscale import UNetDownscale as C_UNet
from model.week5.m2_c_unet_downscale_4 import UNetDownscale as C_UNet_4
from model.week5.m3_swin_unet5 import SwinTransformerSys as SwinTransformerSys1
from model.week5.m3_swin_unet5_hou import SwinTransformerSys as SwinTransformerSys2

from model.week5.m5_kan_unet_downscale import UKAN as UKan
from model.week5.m5_kan_unet_downscale1 import UKAN as UKan2
from model.week5.m5_kan_unet_downscale2 import UKAN as UKan3
from model.week5.m5_kan_unet_downscale3 import UKAN as UKan4 #hou
from model.week5.m5_kan_unet_downscale4 import UKAN as UKan5 #hou

from model.week5.m5_kan_unet1_downscale_hou import UKAN as UKan1 #hou

from model.week5.m6_kan_cunet_downscale2 import UKAN as CUKan
from model.week5.m6_kan_cunet_downscale2_hou import UKAN as CUKan_hou  #hou
from model.week5.m6_kan_cunet1 import UKAN as CUKan1
from model.week5.m6_kan_cunet2 import UKAN as CUKan2
from model.week5.m6_kan_cunet2_1 import UKAN as CUKan2_1
from model.week5.m6_kan_cunet2_1_muti1 import UKAN as CUKan2_1_muti1
from model.week5.m6_kan_cunet2_1_muti2 import UKAN as CUKan2_1_muti2
from model.week5.m6_kan_cunet2_1_muti3 import UKAN as CUKan2_1_muti3
from model.week5.m6_kan_cunet2_1_muti4 import UKAN as CUKan2_1_muti4
from model.week5.m6_kan_cunet2_1_muti5 import UKAN as CUKan2_1_muti5
from model.week5.m6_kan_cunet2_1_muti import UKAN as CUKan2_1_muti
from model.week5.m6_kan_cunet2_1_real import UKAN as CUKan2_1_real
from model.week5.m6_kan_cunet2_1_c1 import UKAN as CUKan2_1_c1
from model.week5.m6_kan_cunet2_1_c3 import UKAN as CUKan2_1_c3
from model.week5.m6_kan_cunet2_1_c4 import UKAN as CUKan2_1_c4
from model.week5.m6_kan_cunet2_1_c5 import UKAN as CUKan2_1_c5
from model.week5.m6_kan_cunet2_kan import UKAN as CUKan2_kan
from model.week5.m6_kan_cunet2_f1 import UKAN as CUKan2_kanf1
from model.week5.m6_kan_cunet2_f2 import UKAN as CUKan2_kanf2
# from model.week5.m6_kan_cunet2_1_1 import UKAN as CUKan2_1_1
from model.week5.m6_kan_cunet2_2_1 import UKAN as CUKan2_2_1
from model.week5.m6_kan_cunet2_2 import UKAN as CUKan2_2
from model.week5.m6_kan_cunet2_3 import UKAN as CUKan2_3
from model.week5.m6_kan_cunet2_4 import UKAN as CUKan2_4
from model.week5.m6_kan_cunet2_5 import UKAN as CUKan2_5
from model.week5.m6_nokan1_cunet2_1 import UKAN as nokan1
from model.week5.m6_kan_cunet3 import UKAN as CUKan3
from model.week5.m6_kan_cunet4 import UKAN as CUKan4
from model.week5.m6_kan_cunet0 import UKAN as CUKan0
from model.week5.m6_kan_cunet_downscale_add_pattern_channel import UKAN as CUKAN_add_pattern_channel
from model.week5.m6_kan_cunet_downscale_add_pattern_channel_view import UKAN as CUKAN_add_pattern_channel_view
from model.week5.m6_kan_cunet_downscale_add_pattern_channel_view_height import UKAN as CUKAN_add_pattern_channel_view_height

from model.image_sr.swinir import SwinIR
from model.image_sr.edsr.edsr import EDSR
from model.image_sr.xunet import XUnet
from  utils.utils  import set_seed

from model.kan_unet.kan_unet import KANU_Net as KANU_Net

from model.week5.FUnet import UKAN as FUnet

from model.week5.m6_kan_cunet1_real import UKAN as CUKAN1_real
from model.week5.m6_kan_cunet2_1_3_real import UKAN as CUKAN2_3_real

from model.week5.m6_kan_cunet0_car import UKAN as CUKAN0_CAF
from model.week5.m6_kan_cunet1_car import UKAN as CUKAN1_CAF
from model.week5.m6_kan_cunet2_1_real_cma import UKAN as CUKAN2_1_real_cma
from model.week5.m6_kan_cunet2_1_real_ec import UKAN as CUKAN2_1_real_ec
from model.week5.m6_kan_cunet2_1_real_cma_t import UKAN as CUKAN2_1_real_cma_t
from model.week5.m6_kan_cunet2_1_real_ec_t import UKAN as CUKAN2_1_real_ec_t
from model.week5.m6_kan_cunet2_1_real_ec_cma_t import UKAN as CUKAN2_1_real_ec_cma_t
from model.week5.m6_kan_cunet2_1_real_kan import UKAN as CUKAN2_1_real_kan
from model.week5.m6_kan_cunet2_1_real_4layer import UKAN   as CUKAN2_1_real_kan_4layer

# from model.tansunet.transunet import VisionTransformer as TransUNet  # 示例
model_classes = {
    "UNet": UNet,
    "UNet_64": UNet,
    "UNet1": UNet1,#hou
    # "UNet2": UNet2,
    # "UNet3": UNet3,

    "C_UNet": C_UNet,
    "C_UNet_4": C_UNet_4,
    "CUkanc1": CUKan2_1_c1,
    "CUkanc3": CUKan2_1_c3,
    "CUkanc4": CUKan2_1_c4,
    "CUkanc5": CUKan2_1_c5,
    "CUkanf1": CUKan2_kanf1,
    "CUkanf2": CUKan2_kanf2,
    "CUKAN_add_pattern_channel": CUKAN_add_pattern_channel,
    "CUKAN_add_pattern_channel_view": CUKAN_add_pattern_channel_view,
    "CUKAN_add_pattern_channel_view_height": CUKAN_add_pattern_channel_view_height,
    # "TransUNet": TransUNet,  # 示例
    "nokan1": nokan1,
    "SwinIR": SwinIR,
    "EDSR": EDSR,
    "XUnet": XUnet,
    "CUKan2_kan": CUKan2_kan,
    "UKan": UKan,
    "UKan2": UKan2,#卷积
    "UKan3": UKan3,#卷积
    "UKan4": UKan4,#卷积
    "UKan5": UKan5,#卷积
    "Ukan1": UKan1,
    "CUKan2_1_muti1": CUKan2_1_muti1,
    "CUKan2_1_muti1_64": CUKan2_1_muti1,
    "CUKan2_1_muti2": CUKan2_1_muti2,
    "CUKan2_1_muti3": CUKan2_1_muti3,
    "CUKan2_1_muti4": CUKan2_1_muti4,
    "CUKan2_1_muti5": CUKan2_1_muti5,
    "CUKan": CUKan,
    "CUKan_hou": CUKan_hou,
    "CUKan1": CUKan1,
    "CUKan2": CUKan2,
    "CUKan2_1": CUKan2_1,
    "CUKan2_1_real": CUKan2_1_real,
    "CUKan2_1_64": CUKan2_1,
    "CUKan2_1_32": CUKan2_1,
    # "CUKan2_1_1": CUKan2_1_1,
    "CUKan2_2": CUKan2_2,
    "CUKan2_2_1": CUKan2_2_1,
    "CUKan2_3": CUKan2_3,
    "CUKan2_4": CUKan2_4,
    "CUKan2_5": CUKan2_5,
    "CUKan0": CUKan0,
    # "CUKan4": CUKan4,
    "CUKAN1_real":CUKAN1_real,
    "CUKAN2_3_real":CUKAN2_3_real,
    "CUKAN0_CAF":CUKAN0_CAF,
    "CUKAN1_CAF":CUKAN1_CAF,
    # "SwinUnet":SwinTransformerSys1,
    # "SwinUnet_hou":SwinTransformerSys2,
    # "CUKAN1_real":CUKAN1_real,
    # "CUKAN2_3_real":CUKAN2_3_real,
    # "CUKAN0_CAF":CUKAN0_CAF,
    # "CUKAN1_CAF":CUKAN1_CAF,
    "CUKAN2_1_real_cma":CUKAN2_1_real_cma,
    "CUKAN2_1_real_ec":CUKAN2_1_real_ec,
    "CUKAN2_1_real_cma_t":CUKAN2_1_real_cma_t,
    "CUKAN2_1_real_ec_t":CUKAN2_1_real_ec_t,
    "CUKAN2_1_real_ec_cma_t":CUKAN2_1_real_ec_cma_t,
    # "KANU_Net":KANU_Net,
    "FUnet":FUnet,
    "CUKAN2_1_real_kan":CUKAN2_1_real_kan,
    "CUKAN2_1_real_kan_4layer":CUKAN2_1_real_kan_4layer,
    "CUKan2_1_real_layer4_64":CUKAN2_1_real_kan_4layer,
    "CUKan2_1_real_64":CUKan2_1_real,
    "UKan_64": UKan,
    # 添加其他模型类
}

def get_model(model_class_name, **kwargs):
    """
    根据类名动态实例化模型。
    Args:
        model_class_name (str): 模型类名，与配置文件中的 `class_name` 对应。
        **kwargs: 初始化模型所需的参数。

    Returns:
        nn.Module: 实例化的模型对象。
    """


    if model_class_name not in model_classes:
        raise ValueError(f"Unknown model class: {model_class_name}")
    # set_seed(42)    # 设置随机种子
    return model_classes[model_class_name](**kwargs)

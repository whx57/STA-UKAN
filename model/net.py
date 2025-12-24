import torch
import torch.nn as nn
import lpips
import numpy as np
from skimage.metrics import structural_similarity as ssim

class MaskedMSELoss(nn.Module):
    def __init__(self, MASK):
        super(MaskedMSELoss, self).__init__()
        self.MASK = MASK

    def forward(self, prediction, target):
        if prediction.shape != self.MASK.shape:
            bias = prediction.shape[0]
            MASK2 = self.MASK[:bias, :, :]
            diff = prediction[~MASK2] - target[~MASK2]
        else:
            diff = prediction[~self.MASK] - target[~self.MASK]
        loss = torch.mean(diff ** 2)
        return loss
class ExtremeWeightedMSE(nn.Module):
    """
    err > thr 时乘以 (1 + alpha) 的权重；其余栅格权重 = 1
    thr 单位与你的数据一致（°C 或 K）；alpha 越大 → 极端越被强化
    """
    def __init__(self, mask: torch.Tensor, thr=4.0, alpha=6.0):
        super().__init__()
        self.register_buffer('mask', mask.bool())
        self.thr   = thr
        self.alpha = alpha

    def forward(self, pred, tgt):
        m   = self.mask.to(pred.device)
        err = pred - tgt
        w   = 1.0 + self.alpha * (err.abs() > self.thr).float()
        loss = (w * err.pow(2)).masked_select(~m).mean()
        return loss


# ----- 小包装：组合两个损失 -----
class ComboLoss(nn.Module):
    def __init__(self,
                 mask: torch.Tensor,
                 w_char=0.7, w_ext=0.3,
                 charbonnier_eps=1e-3,
                 thr=4.0, alpha=6.0):
        super().__init__()
        self.w_char = w_char
        self.w_ext  = w_ext
        self.charb  = MaskedCharbonnierLoss(mask, charbonnier_eps)
        self.ext    = ExtremeWeightedMSE(mask, thr, alpha)

    def forward(self, pred, tgt):
        return (self.w_char * self.charb(pred, tgt) +
                self.w_ext  * self.ext(pred, tgt))
class MaskedCharbonnierLoss(nn.Module):
    def __init__(self, MASK, epsilon=1e-3):
        super(MaskedCharbonnierLoss, self).__init__()
        self.MASK = MASK
        self.epsilon = epsilon

    def forward(self, prediction, target):
        MASK = self.MASK.to(prediction.device)

        if prediction.shape != MASK.shape:
            bias = prediction.shape[0]
            MASK2 = MASK[:bias, :, :]
            if MASK2.shape != prediction.shape:# MASK2 shape torch.Size([1, 192, 288]) does not match prediction shape torch.Size([16, 1, 192, 288])
                #
                MASK2 = MASK[:bias, :, :].unsqueeze(1)  # 扩展维度以匹配预测形状
                # raise ValueError(f"MASK2 shape {MASK2.shape} does not match prediction shape {prediction.shape}")
                
        else:
            MASK2 = MASK

        if MASK2.dtype != torch.bool:
            raise TypeError(f"MASK2 dtype {MASK2.dtype} is not bool")
        # print(prediction.shape)
        # print(target.shape)
        # 使用 masked_select
        diff = prediction.masked_select(~MASK2) - target.masked_select(~MASK2)
        loss = torch.mean(torch.sqrt(diff ** 2 + self.epsilon ** 2))
        return loss


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

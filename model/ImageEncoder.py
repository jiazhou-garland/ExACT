import torch
import torch.nn as nn
from clip.model import LayerNorm
class ImageEncoder(nn.Module):
    def __init__(self, cfg, clip_model):
        super().__init__()
        self.visual = clip_model.visual
        self.conv1 = self.visual.conv1
        self.class_embedding = self.visual.class_embedding
        self.positional_embedding = self.visual.positional_embedding
        self.ln_pre =  self.visual.ln_pre
        self.ln_post = self.visual.ln_post
        self.proj = self.visual.proj

        self.width = 768
        self.output_dim = 512
        scale = self.width ** -0.5
        self.low_level_idx = cfg['MODEL']['EventEncoder']['Low_level_feature_idx']
        self.low_ln = nn.ModuleList()
        self.proj_low_level = []
        for i in range(len(self.low_level_idx)):
            self.low_ln.append(LayerNorm(self.width))
            self.proj_low_level.append(nn.Parameter(scale * torch.randn(self.width, self.output_dim)))

    def forward(self, x: torch.Tensor):
        device = x.device
        x = self.conv1(x)  # shape = [batch, width, H, W]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [batch, width, HW]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat(
            [self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
             x], dim=1)  # shape = [batch, HW + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND

        im_low_level_feature = []
        for i, blk in enumerate(self.visual.transformer.resblocks):
            x = blk(x)
            # extract low-level image feature
            if i + 1 in self.low_level_idx:
                im_low_level_feature.append(x)

        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_post(x[:, 0, :])
        if self.proj is not None:
            x = x @ self.proj

        im_low_level_cls = []
        if len(self.low_level_idx) != 0:
            for i in range(len(im_low_level_feature)):
                im_low_level_cls_i = im_low_level_feature[i].permute(1, 0, 2)
                im_low_level_cls_i = self.low_ln[i](im_low_level_cls_i[:, 0, :])
                im_low_level_cls_i = im_low_level_cls_i @ self.proj_low_level[i].to(device)
                im_low_level_cls.append(im_low_level_cls_i)
            return x, im_low_level_cls
        else:
            return x

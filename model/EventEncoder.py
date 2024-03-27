import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange
from typing import Tuple
from os.path import join, dirname, isfile
import math,tqdm
from .utils.yaml import read_yaml
from .utils.clip_utils import load_clip_to_cpu
from clip.model import QuickGELU,LayerNorm
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from operator import mul
from functools import reduce
_tokenizer = _Tokenizer()

class Attention(nn.Module):
    '''
    A generalized attention module with more flexibility.
    '''

    def __init__(
            self, q_in_dim: int, k_in_dim: int, v_in_dim: int,
            qk_proj_dim: int, v_proj_dim: int, num_heads: int,
            out_dim: int
    ):
        super().__init__()

        self.q_proj = nn.Linear(q_in_dim, qk_proj_dim)
        self.k_proj = nn.Linear(k_in_dim, qk_proj_dim)
        self.v_proj = nn.Linear(v_in_dim, v_proj_dim)
        self.out_proj = nn.Linear(v_proj_dim, out_dim)

        self.num_heads = num_heads
        assert qk_proj_dim % num_heads == 0 and v_proj_dim % num_heads == 0

        self._initialize_weights()

    def _initialize_weights(self):
        for m in (self.q_proj, self.k_proj, self.v_proj, self.out_proj):
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0.)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        assert q.ndim == 3 and k.ndim == 3 and v.ndim == 3
        N = q.size(0);
        assert k.size(0) == N and v.size(0) == N
        Lq, Lkv = q.size(1), k.size(1);
        assert v.size(1) == Lkv

        q, k, v = self.q_proj(q), self.k_proj(k), self.v_proj(v)

        H = self.num_heads
        Cqk, Cv = q.size(-1) // H, v.size(-1) // H

        q = q.view(N, Lq, H, Cqk)
        k = k.view(N, Lkv, H, Cqk)
        v = v.view(N, Lkv, H, Cv)

        aff = torch.einsum('nqhc,nkhc->nqkh', q / (Cqk ** 0.5), k)
        aff = aff.softmax(dim=-2)
        mix = torch.einsum('nqlh,nlhc->nqhc', aff, v)

        out = self.out_proj(mix.flatten(-2))

        return out

class InsertEventPrompt(nn.Module):
    def __init__(
            self, cfg,
            # model def
            feature_dim: int = 768,
            patch_size: Tuple[int, int] = (16, 16),
            num_heads: int = 12,
            in_feature_dim: int = 768,
            qkv_dim: int = 768,
    ):
        super().__init__()
        self.use_event_modality_prompts = cfg['MODEL']['EventEncoder']['use_event_modality_prompts']
        self.num_event_modality_prompts = cfg['MODEL']['EventEncoder']['num_event_modality_prompts']
        if self.use_event_modality_prompts:
            self.event_modality_prompts = nn.Parameter(torch.zeros(self.num_event_modality_prompts, feature_dim))
            self._initialize_event_modality_prompts(patch_size, feature_dim)

        self.use_cross_frame_prompts = cfg['MODEL']['EventEncoder']['use_cross_frame_prompts']
        # for both cross_frame_prompts and intra_frame_prompts we need the cls_proj layer and the num_frames
        if self.use_cross_frame_prompts:
            self.cls_proj = nn.Linear(in_feature_dim, in_feature_dim)

        # for cross_frame_prompts we need a layer norm and attention
        if self.use_cross_frame_prompts:
            self.cross_frame_prompts_ln = LayerNorm(in_feature_dim)
            self.cross_frame_prompts_attn_layer = Attention(
                q_in_dim=in_feature_dim, k_in_dim=in_feature_dim, v_in_dim=in_feature_dim,
                qk_proj_dim=qkv_dim, v_proj_dim=qkv_dim, num_heads=num_heads, out_dim=in_feature_dim)

    def _initialize_event_modality_prompts(self, patch_size, prompt_dim):
        val = math.sqrt(6. / float(3 * reduce(mul, patch_size, 1) + prompt_dim))
        # xavier_uniform initialization
        nn.init.uniform_(self.event_modality_prompts.data, -val, val)

    def forward(self, x, B, T):
        device = x.device
        shape = 0
        if self.use_event_modality_prompts:
            event_modality_prompts = self.event_modality_prompts.expand(B * T, -1, -1).to(device)
            # add global_prompts after the cls token while in front of the original token.
            x = torch.cat((x[:, :1, :], event_modality_prompts, x[:, 1:, :]), dim=1)

        if self.use_cross_frame_prompts:
            BT, N, C = x.shape
            B = BT // T

            cls_token = x[:, 0, :].view(B, T, C)
            cls_token_proj = self.cls_proj(cls_token)  # B, T, C

            # then apply ln and attn if cross_frame_prompts being used
            cross_frame_prompts_norm = self.cross_frame_prompts_ln(cls_token_proj).to(device)
            cross_frame_prompts_attn = cls_token_proj + self.cross_frame_prompts_attn_layer(
                cross_frame_prompts_norm, cross_frame_prompts_norm,
                cross_frame_prompts_norm)
            cross_frame_prompts_attn_reshape = cross_frame_prompts_attn.view(BT, 1, C)
            x = torch.cat([x, cross_frame_prompts_attn_reshape], dim=1)

        return x, shape

class ValueLayer(nn.Module):
    def __init__(self, mlp_layers, activation=nn.ReLU(), num_channels=9):
        assert mlp_layers[-1] == 1, "Last layer of the mlp must have 1 input channel."
        assert mlp_layers[0] == 1, "First layer of the mlp must have 1 output channel"

        nn.Module.__init__(self)
        self.mlp = nn.ModuleList()
        self.activation = activation

        # create mlp
        in_channels = 1
        for out_channels in mlp_layers[1:]:
            self.mlp.append(nn.Linear(in_channels, out_channels))
            in_channels = out_channels

        # init with trilinear kernel
        path = join(dirname(__file__), "quantization_layer_init", "trilinear_init.pth")
        if isfile(path):
            state_dict = torch.load(path)
            self.load_state_dict(state_dict)
        else:
            self.init_kernel(num_channels)

    def forward(self, x):
        # create sample of batchsize 1 and input channels 1
        x = x[None, ..., None]

        # apply mlp convolution
        for i in range(len(self.mlp[:-1])):
            x = self.activation(self.mlp[i](x))

        x = self.mlp[-1](x)
        x = x.squeeze()

        return x

    def init_kernel(self, num_channels):
        ts = torch.zeros((1, 2000))
        optim = torch.optim.Adam(self.parameters(), lr=1e-2)

        torch.manual_seed(1)

    def trilinear_kernel(self, ts, num_channels):
        gt_values = torch.zeros_like(ts)

        gt_values[ts > 0] = (1 - (num_channels - 1) * ts)[ts > 0]
        gt_values[ts < 0] = ((num_channels - 1) * ts + 1)[ts < 0]

        gt_values[ts < -1.0 / (num_channels - 1)] = 0
        gt_values[ts > 1.0 / (num_channels - 1)] = 0

        return gt_values

class QuantizationLayer(nn.Module):
    def __init__(self, dim=15,
                 mlp_layers=[1, 30, 30, 1],
                 activation=nn.LeakyReLU(negative_slope=0.1)):
        nn.Module.__init__(self)
        self.value_layer = ValueLayer(mlp_layers,
                                      activation=activation,
                                      num_channels=dim)
        self.T = int(dim / 3)
        self.dim = dim

    def forward(self, events):
        # get values for each channel
        x, y, t, p = events.t() # b for batch size, 0 to b-1
        C = self.dim
        H, W = int(x.max().cpu().numpy())+1, int(y.max().cpu().numpy())+1

        num_voxels = int(2*C*H*W)
        vox = events[0].new_full([num_voxels,], fill_value=0)

        # normalizing timestamps
        # for bi in range(B):
        #     t[events[:,-1] == bi] /= t[events[:,-1] == bi].max()
        t = t.long() / torch.max(t.long())

        idx_before_bins = x.long() \
                          + H * y.long() \
                          + 0 \
                          + W * H * C * p.long()

        for i_bin in range(C):
            values = t * self.value_layer.forward(t-i_bin/(C-1))
            # draw in voxel grid
            idx = idx_before_bins + W * H * i_bin
            vox.put_(idx.long(), values.to(events.dtype), accumulate=True)

        vox = vox.view(2, 3, self.T, H, W) # 2, 3, T, H, W
        vox = torch.cat([vox[0, ...], vox[1, ...]], 2) # 3, T, H, W
        # print(vox.size()
        return vox

class EventEncoder(nn.Module):

    def __init__(
            self, cfg, clip_model,
            # data shape
            input_size: Tuple[int, int] = (224, 224),
            # model def
            feature_dim: int = 768,
            patch_size: Tuple[int, int] = (16, 16),
            num_layers: int = 12,
            in_feature_dim: int = 768,
    ):
        super().__init__()
        # Todo:
        # Note that the QuantizationLayer is not part of the ExAct model architecture,
        # however, I neglected to annotate this line during training so it must be initializazed when loading pre-trained models.
        self.QuantizationLayer = QuantizationLayer()

        self.visual = clip_model.visual
        self.feature_dim = feature_dim
        self.num_patches = np.prod([x // y for x, y in zip(input_size, patch_size)]) + 1

        self.cls_token = nn.Parameter(torch.zeros([feature_dim]))
        self.pos_embed = nn.Parameter(torch.zeros([self.num_patches, feature_dim]))  # zero initialization for pos_embed
        self.time_embed = nn.Parameter(
            torch.zeros([3, feature_dim]))  # zero initialization for time_embed
        self.visual.proj = clip_model.visual.proj

        self.use_temporal_encoding = cfg['MODEL']['EventEncoder']['use_temporal_encoding']
        self.use_event_modality_prompts = cfg['MODEL']['EventEncoder']['use_event_modality_prompts']
        self.num_event_modality_prompts = cfg['MODEL']['EventEncoder']['num_event_modality_prompts']

        self.InsertEventPrompt = nn.ModuleList()
        for i in range(num_layers):
            self.InsertEventPrompt.append(InsertEventPrompt(cfg))

        self.use_cross_frame_prompts = cfg['MODEL']['EventEncoder']['use_cross_frame_prompts']

        # for low_level_feature projection
        self.width = 768
        self.output_dim = 512
        scale = self.width ** -0.5
        self.low_level_idx = cfg['MODEL']['EventEncoder']['Low_level_feature_idx']
        self.proj_low_level = []
        self.ln_low_level = nn.ModuleList()
        for i in range(len(self.low_level_idx)):
            self.ln_low_level.append(LayerNorm(in_feature_dim))
            self.proj_low_level.append(nn.Parameter(scale * torch.randn(self.width, self.output_dim)))

        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.time_embed, std=0.02)

    def temporal_encoding(self, x, T, B):
        ## Time Embeddings
        x = rearrange(x, '(b t) n m -> (b n) t m', b=B, t=T)

        ## Resizing time embeddings in case they don't match
        if T != self.time_embed.size(0):
            time_embed = self.time_embed.unsqueeze(0).transpose(1, 2)
            new_time_embed = F.interpolate(time_embed, size=(T), mode='nearest')
            new_time_embed = new_time_embed.transpose(1, 2).squeeze(0)
            x = x + new_time_embed
        else:
            x = x + self.time_embed

        x = rearrange(x, '(b n) t m -> (b t) n m', b=B, t=T)
        return x

    def forward(self, x, actual_event_length):
        # print(x.size())
        B, C, T, H, W = x.size()
        x = x.permute(0, 2, 1, 3, 4).flatten(0, 1)  # BT, C, H, W
        # x transform into batches
        x = self.visual.conv1(x)  # shape = [BT, width, H, W]
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)  # shape = [BT, HW, width]
        # add cls token
        x = torch.cat(
            [self.visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype,
                                                                   device=x.device), x], dim=1)  # BT, HW+1, width
        # add pos_embed
        x = x + self.visual.positional_embedding.to(x.dtype)
        # add temporal_embed
        if self.use_temporal_encoding:
            x = self.temporal_encoding(x, T, B)
        # layer normalization
        x = self.visual.ln_pre(x)

        for i, blk in enumerate(self.visual.transformer.resblocks):
            # the global prompts are inserted between every transformer block,
            # by concatenating with the x output from the last transformer block.
            x, shape = self.InsertEventPrompt[i](x, B, T)

            x = x.permute(1, 0, 2)  # NLD -> LND
            x = blk(x)
            x = x.permute(1, 0, 2)  # NLD -> LND

            # extract the output x without the cross_frame_prompts
            if self.use_cross_frame_prompts:
                x = x[:, :-1, :]
            # extract the output x without the event_modality_prompts
            if self.use_event_modality_prompts:
                x = torch.cat((x[:, :1, :], x[:, self.num_event_modality_prompts + 1:, :]), dim=1)

        # extract cls token for the final event embedding
        cls_x = self.visual.ln_post(x[:, 0, :])
        cls_x = cls_x @ self.visual.proj  # b, t, e

        # average cls tokens from all actual frames without the padded ones.
        cls_x = rearrange(cls_x, '(b t) e -> b t e', b=B, t=T)  # b,T,512

        return cls_x

if __name__ == '__main__':
    cfg = read_yaml('F:\code\HumanECLIP\Configs/backbone_ddp.yaml')
    clip_model = load_clip_to_cpu(cfg).float()
    EventEncoder = EventEncoder(cfg,clip_model).float()

    test = torch.ones((2,3,5,224,224)) # B, C, T, H, W
    cls_x = EventEncoder(test)
    print(cls_x.shape)

    # for name, param in EventEncoder.named_parameters():
    #     print(name)


import torch
import torch.nn as nn
from collections import OrderedDict
from timm.models.layers import DropPath
from .LossFunction import symmetric_cross_entropy_loss
from .utils.yaml import read_yaml

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class Concept_Project_function(nn.Module):
    def __init__(self, input_dim, cfg):
        super().__init__()
        concept_dim = cfg['MODEL']['Conceptualizing_Alignment']['concept_dim']
        num_concept = cfg['MODEL']['Conceptualizing_Alignment']['num_concept']
        self.Concept_Project_function = nn.Sequential(OrderedDict([
            ("linear1", nn.Linear(input_dim, input_dim)),
            ("relu", QuickGELU()),
            ("linear2", nn.Linear(input_dim , concept_dim*num_concept))
        ]))
        if cfg['Trainer']['Precision'] == "fp16":
            self.Concept_Project_function.half()  # float32 -> float16

    def forward(self, f):
        device = f.device
        projected_concept = self.Concept_Project_function(f).to(device)  # (batch, out_dim)
        return projected_concept

class Concept_Fusion_function(nn.Module):
    def __init__(self, input_dim, cfg):
        super().__init__()
        concept_dim = cfg['MODEL']['Conceptualizing_Alignment']['concept_dim']
        num_concept = cfg['MODEL']['Conceptualizing_Alignment']['num_concept']
        self.Concept_Project_function = nn.Sequential(OrderedDict([
            ("linear1", nn.Linear(input_dim, input_dim // 2)),
            ("relu", QuickGELU()),
            ("linear2", nn.Linear(input_dim // 2, input_dim // 2))
        ]))
        if cfg['Trainer']['Precision'] == "fp16":
            self.Concept_Project_function.half()  # float32 -> float16

    def forward(self, f):
        device = f.device
        projected_concept = self.Concept_Project_function(f).to(device)  # (batch, out_dim)
        return projected_concept

class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class DisAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.mu_proj = nn.Linear(int(dim/2), dim)
        self.mu_proj_drop = nn.Dropout(proj_drop)
        self.logsig_proj = nn.Linear(int(dim/2), dim)
        self.logsig_proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x.reshape(B*N,C))
        qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
         # (3, B, mu_heads_num+logsig_heads_num, N, dim_heads)
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            attn = attn + mask
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C).reshape(B, N, 2, int(C/2))

        mu = x[:,:,0,:] # B, N, C/2
        logsigma = x[:,:,1,:] # B, N, C/2
        mu = self.mu_proj(mu) # B, N, C
        mu = self.mu_proj_drop(mu) # B, N, C
        logsigma = self.logsig_proj(logsigma)  # B, N, C
        logsigma = self.logsig_proj_drop(logsigma) # B, N, C
        return mu, logsigma, attn

class DisTrans(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.1,
        attn_drop=0.1,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.fc = nn.Linear(dim, dim)
        self.act = act_layer()
        self.norm1 = norm_layer(dim)
        self.attn = DisAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.norm3 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mu_mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )
        self.logsig_mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x, mask=None):
        x_ = self.norm1(self.act(self.fc(x)))
        mu, logsigma, attn = self.attn(x_, mask=mask)
        mu = x + self.drop_path(mu)
        mu = mu + self.drop_path(self.mu_mlp(self.norm2(mu)))
        logsigma = logsigma + self.drop_path(self.logsig_mlp(self.norm3(logsigma)))
        return mu, logsigma, attn

# ori
class Conceptualizing_Alignment1(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.num_concept = cfg['MODEL']['Conceptualizing_Alignment']['num_concept']
        self.concept_dim = cfg['MODEL']['Conceptualizing_Alignment']['concept_dim']
        self.sample_num = cfg['MODEL']['Conceptualizing_Alignment']['sample_num']
        self.loss_sigma_threshold = cfg['LossFunction']['loss_sigma_threshold']
        self.txt_Concept = Concept_Project_function(512, cfg)
        self.ev_Concept = Concept_Project_function(512, cfg)
        self.txt_Concept_fusion = nn.ModuleList()
        self.ev_Concept_fusion = nn.ModuleList()
        for i in range(self.num_concept // 2):
            self.txt_Concept_fusion.append(Concept_Fusion_function(self.num_concept*self.concept_dim//pow(2,i), cfg))
            self.ev_Concept_fusion.append(Concept_Fusion_function(self.num_concept*self.concept_dim//pow(2,i), cfg))
        self.ev_gau_encoder = DisTrans(self.concept_dim, 8)
        self.txt_gau_encoder = DisTrans(self.concept_dim, 8)
        if cfg['Trainer']['Precision'] == "fp16":
            self.ev_gau_encoder.half()
            self.txt_gau_encoder.half()

    def Content_Uncertainty(self, ev_concept, txt_concept):

        # [step3] Content Uncertainty:
        ev_mu, ev_logsigma, _ = self.ev_gau_encoder(ev_concept) #B, num_concept, concept_dim
        txt_mu, txt_logsigma, _ = self.txt_gau_encoder(txt_concept) #B, num_concept, concept_dim

        return ev_mu, ev_logsigma, txt_mu, txt_logsigma

    def Point_Feature_Sampling(self, ev_mu, ev_logsigma, txt_mu, txt_logsigma):
        # [step4] Point Feature Sampling:
        # ev = [ev_mu] * self.sample_num
        # txt = [txt_mu] * self.sample_num
        ev = []
        txt = []
        for i in range(self.sample_num):
            eps = torch.randn(ev_mu.shape[0], ev_mu.shape[1], ev_mu.shape[2], device=ev_mu.device)
            ev_i = ev_mu + torch.exp(ev_logsigma) * eps # B, num_concept, concept_dim
            ev.append(ev_i)

            eps = torch.randn(txt_mu.shape[0], txt_mu.shape[1], txt_mu.shape[2], device=txt_mu.device)
            txt_i = txt_mu + torch.exp(txt_logsigma) * eps # B, num_concept, concept_dim
            txt.append(txt_i)

        ev_embeds = torch.stack(ev) #2*self.sample_num B, num_concept, concept_dim
        txt_embeds = torch.stack(txt) #2*self.sample_num B, num_concept, concept_dim

        return ev_embeds, txt_embeds

    def forward(self, ev_f, txt_f, actual_event_length, logit_scale):
        """
        ev_f: extracted event feature, B,T,dim
        txt_f: extracted text feature, B,dim
        """
        device = txt_f.device
        # print(device)

        # # [step1] Concept Extraction:
        B_txt, txt_dim = txt_f.shape
        # txt_concept = self.txt_Concept(txt_f).reshape(B_txt, self.num_concept, self.concept_dim) # B_txt, num_concept, concept_dim
        #
        B_ev, T, ev_dim = ev_f.shape
        # ev_concept = []
        # for i in range(T):
        #     ev_f_i = ev_f[:,i,:] # B, ev_dim
        #     ev_f_i = self.ev_Concept(ev_f_i).reshape(B_ev, self.num_concept, self.concept_dim) # B_ev, T, num_concept, concept_dim
        #     ev_concept.append(ev_f_i)
        # ev_concept = torch.stack(ev_concept).to(device) # T, B_ev, num_concept, concept_dim
        #
        # # [step2] Event Concept fusion: fusion event frames by text concepts
        # T_weight = torch.einsum('abde,cde->abc', [ev_concept, txt_concept]) # T, B_ev, B_txt
        # T_weight = torch.mean(T_weight, dim=2) # T, B_ev
        # for i in range(B_ev):
        #     T_weight[actual_event_length[i]:,:] = 0 # exclude the padded event frames
        # T_weight = torch.softmax(T_weight, dim=0)  # T, B_ev
        # ev_concept = torch.einsum('abcd,ab->bcd', [ev_concept, T_weight]) # B_ev, num_concept, concept_dim
        #
        # # [step6] Progressive Progress:
        # txt_concept_flatten = txt_concept.reshape(B_txt, self.num_concept*self.concept_dim)
        # ev_concept_flatten = ev_concept.reshape(B_ev, self.num_concept*self.concept_dim)
        # ev_concept_pro = [ev_concept]
        # txt_concept_pro = [txt_concept]
        num_concept = self.num_concept
        # # for i in range(self.num_concept // 2 - 1):
        # #     txt_concept_i = self.txt_Concept_fusion[i](txt_concept_flatten).reshape(B_txt, self.num_concept//pow(2,i+1), self.concept_dim)
        # #     ev_concept_i = self.ev_Concept_fusion[i](ev_concept_flatten).reshape(B_ev, self.num_concept//pow(2,i+1), self.concept_dim)
        # #     ev_concept_pro.append(ev_concept_i)
        # #     txt_concept_pro.append(txt_concept_i)
        # #     txt_concept_flatten = txt_concept_i.reshape(B_txt, self.num_concept*self.concept_dim//pow(2,i+1))
        # #     ev_concept_flatten = ev_concept_i.reshape(B_ev, self.num_concept*self.concept_dim//pow(2,i+1))
        # #     num_concept += self.num_concept//pow(2,i+1)
        # #
        # ev_concept_pro = torch.cat(ev_concept_pro, dim=1).to(device)# B_ev, num_concept, concept_dim
        # txt_concept_pro = torch.cat(txt_concept_pro, dim=1).to(device)# B_txt, num_concept, concept_dim

        # [step3] Content Uncertainty:
        ev_concept_pro = torch.mean(ev_f, dim=1).reshape(num_concept, B_ev, -1).permute(1,0,2) # B_ev, num_concept, concept_dim
        txt_concept_pro = txt_f.reshape(num_concept, B_txt, -1).permute(1,0,2) # B_txt, num_concept, concept_dim

        ev_mu, ev_logsigma, txt_mu, txt_logsigma = self.Content_Uncertainty(ev_concept_pro, txt_concept_pro) # B_txt / B_ev, num_concept, concept_dim

        # print(ev_mu.mean())
        # print(ev_mu)
        # print(txt_mu)
        # print(ev_logsigma.mean())
        # print(ev_mu.shape)
        # print(ev_logsigma)
        # print(txt_logsigma)
        # [step4] Point Feature Sampling:
        ev_embeds, txt_embeds = \
            self.Point_Feature_Sampling(ev_mu, ev_logsigma, txt_mu, txt_logsigma)  # sample_num, B, num_concept, concept_dim
        # loss_SmoothL1Loss_ev, loss_SmoothL1Loss_txt = torch.tensor(0.0).to(device), torch.tensor(0.0).to(device)
        loss_SmoothL1Loss_ev = torch.nn.SmoothL1Loss()(ev_embeds,ev_concept_pro.repeat(self.sample_num,1,1,1))
        loss_SmoothL1Loss_txt = torch.nn.SmoothL1Loss()(txt_embeds,txt_concept_pro.repeat(self.sample_num,1,1,1))

        # print(ev_embeds_.mean())
        # print(ev_embeds_)
        # print(txt_embeds)
        # print(ev_embeds_.shape)
        ev_embeds_ = ev_embeds.permute(0,2,1,3).reshape(num_concept*self.sample_num, B_ev, -1)
        txt_embeds_ = txt_embeds.permute(0,2,1,3).reshape(num_concept*self.sample_num, B_txt, -1)

        # [step5] Calculate Loss Function:
        # contrastive loss function
        if self.training:
            loss_cross_entroy = torch.tensor(0.0).to(device)
            for i in range(num_concept*self.sample_num):
                # the contrastive loss is only calculated for every concept.
                logits = logit_scale * ev_embeds_[i,:,:] @ txt_embeds_[i,:,:].t()
                loss_cross_entroy += symmetric_cross_entropy_loss(logits)
                # print(loss_cross_entroy)
                # print(logits)

            ev_f = ev_f.mean(1)
            logits_ori = logit_scale * ev_f @ txt_f.t()
            loss_cross_entroy_ori = symmetric_cross_entropy_loss(logits_ori)
            # print(loss_cross_entroy)
            # print(logits)

            loss_logsigma_sum, cov_ev_txt_loss, cov_ev_loss, cov_txt_loss = \
                torch.tensor(0.0).to(device), torch.tensor(0.0).to(device), torch.tensor(0.0).to(device), torch.tensor(0.0).to(device)
            # logsima_sum
            logsigma_sum = torch.exp(ev_logsigma).sum() + torch.exp(txt_logsigma).sum()
            # # print(logsigma_sum)
            # # loss_logsigma_sum = torch.tensor(max(0, self.loss_sigma_threshold - logsigma_sum)).to(device)
            loss_logsigma_sum = logsigma_sum.to(device)

            # # cross-correlation matrix
            # ev_embeds_norm = (ev_embeds - ev_embeds.mean(0)) / ev_embeds.std(0) # sample_num, B_ev, num_concept, concept_dim
            # ev_embeds_norm = ev_embeds_norm.mean(dim=0).reshape(num_concept*B_ev, -1) # B_ev*num_concept, concept_dim
            # txt_embeds_norm = (txt_embeds - txt_embeds.mean(0)) / txt_embeds.std(0) # sample_num, B_txt, num_concept, concept_dim
            # txt_embeds_norm = txt_embeds_norm.mean(dim=0).reshape(num_concept*B_txt, -1) # B_txt*num_concept, concept_dim
            # cov_ev_txt = torch.einsum('ab, bc->ac', [ev_embeds_norm, txt_embeds_norm.t()])
            # cov_ev = torch.einsum('ab, bc->ac', [ev_embeds_norm, ev_embeds_norm.t()])
            # cov_txt = torch.einsum('ab, bc->ac', [txt_embeds_norm, txt_embeds_norm.t()])
            # cov_ev_txt_loss = (cov_ev_txt.sum()-2*torch.diagonal(cov_ev_txt).sum())
            # cov_ev_loss = (cov_ev.sum()-2*torch.diagonal(cov_ev).sum())
            # cov_txt_loss = (cov_txt.sum()-2*torch.diagonal(cov_txt).sum())
            # print(cov_ev_txt_loss)
            # print(cov_ev_loss)
            # print(cov_txt_loss)

            # return loss_cross_entroy, loss_logsigma_sum, cov_ev_txt_loss, cov_ev_loss, cov_txt_loss
            return loss_cross_entroy, loss_cross_entroy_ori, loss_logsigma_sum,\
                loss_SmoothL1Loss_ev, loss_SmoothL1Loss_txt, \
                cov_ev_txt_loss, cov_ev_loss, cov_txt_loss

        else:

            # the last concept for Progressive Progress,
            # that represents the highest semantic meaning, is used for evaluation.
            logits = logit_scale * ev_embeds_[-1, :, :] @ txt_embeds_[-1, :, :].t()  # B_ev*B_txt

            ev_f = ev_f.mean(1)
            logits_ori = logit_scale * ev_f @ txt_f.t()  # B_ev*B_txt
            return logits, logits_ori

# +Concept Extraction
class Conceptualizing_Alignment2(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.num_concept = cfg['MODEL']['Conceptualizing_Alignment']['num_concept']
        self.concept_dim = cfg['MODEL']['Conceptualizing_Alignment']['concept_dim']
        self.sample_num = cfg['MODEL']['Conceptualizing_Alignment']['sample_num']
        self.loss_sigma_threshold = cfg['LossFunction']['loss_sigma_threshold']
        self.txt_Concept = Concept_Project_function(512, cfg)
        self.ev_Concept = Concept_Project_function(512, cfg)
        self.txt_Concept_fusion = nn.ModuleList()
        self.ev_Concept_fusion = nn.ModuleList()
        for i in range(self.num_concept // 2):
            self.txt_Concept_fusion.append(Concept_Fusion_function(self.num_concept*self.concept_dim//pow(2,i), cfg))
            self.ev_Concept_fusion.append(Concept_Fusion_function(self.num_concept*self.concept_dim//pow(2,i), cfg))
        self.ev_gau_encoder = DisTrans(self.concept_dim, 8)
        self.txt_gau_encoder = DisTrans(self.concept_dim, 8)
        if cfg['Trainer']['Precision'] == "fp16":
            self.ev_gau_encoder.half()
            self.txt_gau_encoder.half()

    def Content_Uncertainty(self, ev_concept, txt_concept):

        # [step3] Content Uncertainty:
        ev_mu, ev_logsigma, _ = self.ev_gau_encoder(ev_concept) #B, num_concept, concept_dim
        txt_mu, txt_logsigma, _ = self.txt_gau_encoder(txt_concept) #B, num_concept, concept_dim

        return ev_mu, ev_logsigma, txt_mu, txt_logsigma

    def Point_Feature_Sampling(self, ev_mu, ev_logsigma, txt_mu, txt_logsigma):
        # [step4] Point Feature Sampling:
        # ev = [ev_mu] * self.sample_num
        # txt = [txt_mu] * self.sample_num
        ev = []
        txt = []
        for i in range(self.sample_num):
            eps = torch.randn(ev_mu.shape[0], ev_mu.shape[1], ev_mu.shape[2], device=ev_mu.device)
            ev_i = ev_mu + torch.exp(ev_logsigma) * eps # B, num_concept, concept_dim
            ev.append(ev_i)

            eps = torch.randn(txt_mu.shape[0], txt_mu.shape[1], txt_mu.shape[2], device=txt_mu.device)
            txt_i = txt_mu + torch.exp(txt_logsigma) * eps # B, num_concept, concept_dim
            txt.append(txt_i)

        ev_embeds = torch.stack(ev) #2*self.sample_num B, num_concept, concept_dim
        txt_embeds = torch.stack(txt) #2*self.sample_num B, num_concept, concept_dim

        return ev_embeds, txt_embeds

    def forward(self, ev_f, txt_f, actual_event_length, logit_scale):
        """
        ev_f: extracted event feature, B,T,dim
        txt_f: extracted text feature, B,dim
        """
        device = txt_f.device
        # print(device)

        # # [step1] Concept Extraction:
        B_txt, txt_dim = txt_f.shape
        txt_concept = self.txt_Concept(txt_f).reshape(B_txt, self.num_concept, self.concept_dim) # B_txt, num_concept, concept_dim
        #
        B_ev, T, ev_dim = ev_f.shape
        ev_concept = []
        for i in range(T):
            ev_f_i = ev_f[:,i,:] # B, ev_dim
            ev_f_i = self.ev_Concept(ev_f_i).reshape(B_ev, self.num_concept, self.concept_dim) # B_ev, T, num_concept, concept_dim
            ev_concept.append(ev_f_i)
        ev_concept = torch.stack(ev_concept).to(device) # T, B_ev, num_concept, concept_dim
        ev_concept = ev_concept.mean(0)

        # # [step2] Event Concept fusion: fusion event frames by text concepts
        # T_weight = torch.einsum('abde,cde->abc', [ev_concept, txt_concept]) # T, B_ev, B_txt
        # T_weight = torch.mean(T_weight, dim=2) # T, B_ev
        # for i in range(B_ev):
        #     T_weight[actual_event_length[i]:,:] = 0 # exclude the padded event frames
        # T_weight = torch.softmax(T_weight, dim=0)  # T, B_ev
        # ev_concept = torch.einsum('abcd,ab->bcd', [ev_concept, T_weight]) # B_ev, num_concept, concept_dim
        #
        # # [step6] Progressive Progress:
        # txt_concept_flatten = txt_concept.reshape(B_txt, self.num_concept*self.concept_dim)
        # ev_concept_flatten = ev_concept.reshape(B_ev, self.num_concept*self.concept_dim)
        ev_concept_pro = [ev_concept]
        txt_concept_pro = [txt_concept]
        num_concept = self.num_concept
        # # for i in range(self.num_concept // 2 - 1):
        # #     txt_concept_i = self.txt_Concept_fusion[i](txt_concept_flatten).reshape(B_txt, self.num_concept//pow(2,i+1), self.concept_dim)
        # #     ev_concept_i = self.ev_Concept_fusion[i](ev_concept_flatten).reshape(B_ev, self.num_concept//pow(2,i+1), self.concept_dim)
        # #     ev_concept_pro.append(ev_concept_i)
        # #     txt_concept_pro.append(txt_concept_i)
        # #     txt_concept_flatten = txt_concept_i.reshape(B_txt, self.num_concept*self.concept_dim//pow(2,i+1))
        # #     ev_concept_flatten = ev_concept_i.reshape(B_ev, self.num_concept*self.concept_dim//pow(2,i+1))
        # #     num_concept += self.num_concept//pow(2,i+1)
        # #
        ev_concept_pro = torch.cat(ev_concept_pro, dim=1).to(device)# B_ev, num_concept, concept_dim
        txt_concept_pro = torch.cat(txt_concept_pro, dim=1).to(device)# B_txt, num_concept, concept_dim

        # [step3] Content Uncertainty:
        # ev_concept_pro = torch.mean(ev_f, dim=1).reshape(num_concept, B_ev, -1).permute(1,0,2) # B_ev, num_concept, concept_dim
        # txt_concept_pro = txt_f.reshape(num_concept, B_txt, -1).permute(1,0,2) # B_txt, num_concept, concept_dim

        ev_mu, ev_logsigma, txt_mu, txt_logsigma = self.Content_Uncertainty(ev_concept_pro, txt_concept_pro) # B_txt / B_ev, num_concept, concept_dim

        # print(ev_mu.mean())
        # print(ev_mu)
        # print(txt_mu)
        # print(ev_logsigma.mean())
        # print(ev_mu.shape)
        # print(ev_logsigma)
        # print(txt_logsigma)
        # [step4] Point Feature Sampling:
        ev_embeds, txt_embeds = \
            self.Point_Feature_Sampling(ev_mu, ev_logsigma, txt_mu, txt_logsigma)  # sample_num, B, num_concept, concept_dim
        # loss_SmoothL1Loss_ev, loss_SmoothL1Loss_txt = torch.tensor(0.0).to(device), torch.tensor(0.0).to(device)
        loss_SmoothL1Loss_ev = torch.nn.SmoothL1Loss()(ev_embeds,ev_concept_pro.repeat(self.sample_num,1,1,1))
        loss_SmoothL1Loss_txt = torch.nn.SmoothL1Loss()(txt_embeds,txt_concept_pro.repeat(self.sample_num,1,1,1))

        # print(ev_embeds_.mean())
        # print(ev_embeds_)
        # print(txt_embeds)
        # print(ev_embeds_.shape)
        ev_embeds_ = ev_embeds.permute(0,2,1,3).reshape(num_concept*self.sample_num, B_ev, -1)
        txt_embeds_ = txt_embeds.permute(0,2,1,3).reshape(num_concept*self.sample_num, B_txt, -1)

        # [step5] Calculate Loss Function:
        # contrastive loss function
        if self.training:
            loss_cross_entroy = torch.tensor(0.0).to(device)
            for i in range(num_concept*self.sample_num):
                # the contrastive loss is only calculated for every concept.
                logits = logit_scale * ev_embeds_[i,:,:] @ txt_embeds_[i,:,:].t()
                loss_cross_entroy += symmetric_cross_entropy_loss(logits)
                # print(loss_cross_entroy)
                # print(logits)

            ev_f = ev_f.mean(1)
            logits_ori = logit_scale * ev_f @ txt_f.t()
            loss_cross_entroy_ori = symmetric_cross_entropy_loss(logits_ori)
            # print(loss_cross_entroy)
            # print(logits)

            loss_logsigma_sum, cov_ev_txt_loss, cov_ev_loss, cov_txt_loss = \
                torch.tensor(0.0).to(device), torch.tensor(0.0).to(device), torch.tensor(0.0).to(device), torch.tensor(0.0).to(device)
            # logsima_sum
            logsigma_sum = torch.exp(ev_logsigma).sum() + torch.exp(txt_logsigma).sum()
            # # print(logsigma_sum)
            # # loss_logsigma_sum = torch.tensor(max(0, self.loss_sigma_threshold - logsigma_sum)).to(device)
            loss_logsigma_sum = logsigma_sum.to(device)

            # # cross-correlation matrix
            # ev_embeds_norm = (ev_embeds - ev_embeds.mean(0)) / ev_embeds.std(0) # sample_num, B_ev, num_concept, concept_dim
            # ev_embeds_norm = ev_embeds_norm.mean(dim=0).reshape(num_concept*B_ev, -1) # B_ev*num_concept, concept_dim
            # txt_embeds_norm = (txt_embeds - txt_embeds.mean(0)) / txt_embeds.std(0) # sample_num, B_txt, num_concept, concept_dim
            # txt_embeds_norm = txt_embeds_norm.mean(dim=0).reshape(num_concept*B_txt, -1) # B_txt*num_concept, concept_dim
            # cov_ev_txt = torch.einsum('ab, bc->ac', [ev_embeds_norm, txt_embeds_norm.t()])
            # cov_ev = torch.einsum('ab, bc->ac', [ev_embeds_norm, ev_embeds_norm.t()])
            # cov_txt = torch.einsum('ab, bc->ac', [txt_embeds_norm, txt_embeds_norm.t()])
            # cov_ev_txt_loss = (cov_ev_txt.sum()-2*torch.diagonal(cov_ev_txt).sum())
            # cov_ev_loss = (cov_ev.sum()-2*torch.diagonal(cov_ev).sum())
            # cov_txt_loss = (cov_txt.sum()-2*torch.diagonal(cov_txt).sum())
            # print(cov_ev_txt_loss)
            # print(cov_ev_loss)
            # print(cov_txt_loss)

            # return loss_cross_entroy, loss_logsigma_sum, cov_ev_txt_loss, cov_ev_loss, cov_txt_loss
            return loss_cross_entroy, loss_cross_entroy_ori, loss_logsigma_sum,\
                loss_SmoothL1Loss_ev, loss_SmoothL1Loss_txt, \
                cov_ev_txt_loss, cov_ev_loss, cov_txt_loss

        else:

            # the last concept for Progressive Progress,
            # that represents the highest semantic meaning, is used for evaluation.
            logits = logit_scale * ev_embeds_[-1, :, :] @ txt_embeds_[-1, :, :].t()  # B_ev*B_txt

            ev_f = ev_f.mean(1)
            logits_ori = logit_scale * ev_f @ txt_f.t()  # B_ev*B_txt
            return logits, logits_ori

#+Concept Extraction+Event Concept fusion
class Conceptualizing_Alignment3(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.num_concept = cfg['MODEL']['Conceptualizing_Alignment']['num_concept']
        self.concept_dim = cfg['MODEL']['Conceptualizing_Alignment']['concept_dim']
        self.sample_num = cfg['MODEL']['Conceptualizing_Alignment']['sample_num']
        self.loss_sigma_threshold = cfg['LossFunction']['loss_sigma_threshold']
        self.txt_Concept = Concept_Project_function(512, cfg)
        self.ev_Concept = Concept_Project_function(512, cfg)
        self.txt_Concept_fusion = nn.ModuleList()
        self.ev_Concept_fusion = nn.ModuleList()
        for i in range(self.num_concept // 2):
            self.txt_Concept_fusion.append(Concept_Fusion_function(self.num_concept*self.concept_dim//pow(2,i), cfg))
            self.ev_Concept_fusion.append(Concept_Fusion_function(self.num_concept*self.concept_dim//pow(2,i), cfg))
        self.ev_gau_encoder = DisTrans(self.concept_dim, 8)
        self.txt_gau_encoder = DisTrans(self.concept_dim, 8)
        if cfg['Trainer']['Precision'] == "fp16":
            self.ev_gau_encoder.half()
            self.txt_gau_encoder.half()

    def Content_Uncertainty(self, ev_concept, txt_concept):

        # [step3] Content Uncertainty:
        ev_mu, ev_logsigma, _ = self.ev_gau_encoder(ev_concept) #B, num_concept, concept_dim
        txt_mu, txt_logsigma, _ = self.txt_gau_encoder(txt_concept) #B, num_concept, concept_dim

        return ev_mu, ev_logsigma, txt_mu, txt_logsigma

    def Point_Feature_Sampling(self, ev_mu, ev_logsigma, txt_mu, txt_logsigma):
        # [step4] Point Feature Sampling:
        # ev = [ev_mu] * self.sample_num
        # txt = [txt_mu] * self.sample_num
        ev = []
        txt = []
        for i in range(self.sample_num):
            eps = torch.randn(ev_mu.shape[0], ev_mu.shape[1], ev_mu.shape[2], device=ev_mu.device)
            ev_i = ev_mu + torch.exp(ev_logsigma) * eps # B, num_concept, concept_dim
            ev.append(ev_i)

            eps = torch.randn(txt_mu.shape[0], txt_mu.shape[1], txt_mu.shape[2], device=txt_mu.device)
            txt_i = txt_mu + torch.exp(txt_logsigma) * eps # B, num_concept, concept_dim
            txt.append(txt_i)

        ev_embeds = torch.stack(ev) #self.sample_num B, num_concept, concept_dim
        txt_embeds = torch.stack(txt) #self.sample_num B, num_concept, concept_dim

        return ev_embeds, txt_embeds

    def forward(self, ev_f, txt_f, actual_event_length, logit_scale):
        """
        ev_f: extracted event feature, B,T,dim
        txt_f: extracted text feature, B,dim
        """
        device = txt_f.device
        # print(device)

        # # [step1] Concept Extraction:
        B_txt, txt_dim = txt_f.shape
        txt_concept = self.txt_Concept(txt_f).reshape(B_txt, self.num_concept, self.concept_dim) # B_txt, num_concept, concept_dim
        #
        B_ev, T, ev_dim = ev_f.shape
        ev_concept = []
        for i in range(T):
            ev_f_i = ev_f[:,i,:] # B, ev_dim
            ev_f_i = self.ev_Concept(ev_f_i).reshape(B_ev, self.num_concept, self.concept_dim) # B_ev, T, num_concept, concept_dim
            ev_concept.append(ev_f_i)
        ev_concept = torch.stack(ev_concept).to(device) # T, B_ev, num_concept, concept_dim
        #
        # # [step2] Event Concept fusion: fusion event frames by text concepts
        T_weight = torch.einsum('abde,cde->abc', [ev_concept, txt_concept]) # T, B_ev, B_txt
        T_weight = torch.mean(torch.mean(T_weight, dim=2), dim=1) # T
        # for i in range(B_ev):
        #     T_weight[actual_event_length[i]:,:] = 0.0 # exclude the padded event frames
        T_weight = torch.softmax(T_weight, dim=0)  # T
        ev_concept = torch.einsum('abcd,a->bcd', [ev_concept, T_weight]) # B_ev, num_concept, concept_dim
        #
        # # [step6] Progressive Progress:
        # txt_concept_flatten = txt_concept.reshape(B_txt, self.num_concept*self.concept_dim)
        # ev_concept_flatten = ev_concept.reshape(B_ev, self.num_concept*self.concept_dim)
        ev_concept_pro = [ev_concept]
        txt_concept_pro = [txt_concept]
        num_concept = self.num_concept
        # # for i in range(self.num_concept // 2 - 1):
        # #     txt_concept_i = self.txt_Concept_fusion[i](txt_concept_flatten).reshape(B_txt, self.num_concept//pow(2,i+1), self.concept_dim)
        # #     ev_concept_i = self.ev_Concept_fusion[i](ev_concept_flatten).reshape(B_ev, self.num_concept//pow(2,i+1), self.concept_dim)
        # #     ev_concept_pro.append(ev_concept_i)
        # #     txt_concept_pro.append(txt_concept_i)
        # #     txt_concept_flatten = txt_concept_i.reshape(B_txt, self.num_concept*self.concept_dim//pow(2,i+1))
        # #     ev_concept_flatten = ev_concept_i.reshape(B_ev, self.num_concept*self.concept_dim//pow(2,i+1))
        # #     num_concept += self.num_concept//pow(2,i+1)
        # #
        ev_concept_pro = torch.cat(ev_concept_pro, dim=1).to(device)# B_ev, num_concept, concept_dim
        txt_concept_pro = torch.cat(txt_concept_pro, dim=1).to(device)# B_txt, num_concept, concept_dim

        # [step3] Content Uncertainty:
        # ev_concept_pro = torch.mean(ev_f, dim=1).reshape(num_concept, B_ev, -1).permute(1,0,2) # B_ev, num_concept, concept_dim
        # txt_concept_pro = txt_f.reshape(num_concept, B_txt, -1).permute(1,0,2) # B_txt, num_concept, concept_dim

        ev_mu, ev_logsigma, txt_mu, txt_logsigma = self.Content_Uncertainty(ev_concept_pro, txt_concept_pro) # B_txt / B_ev, num_concept, concept_dim

        # print(ev_mu.mean())
        # print(ev_mu)
        # print(txt_mu)
        # print(ev_logsigma.mean())
        # print(ev_mu.shape)
        # print(ev_logsigma)
        # print(txt_logsigma)
        # [step4] Point Feature Sampling:
        ev_embeds, txt_embeds = \
            self.Point_Feature_Sampling(ev_mu, ev_logsigma, txt_mu, txt_logsigma)  # sample_num, B, num_concept, concept_dim
        # loss_SmoothL1Loss_ev, loss_SmoothL1Loss_txt = torch.tensor(0.0).to(device), torch.tensor(0.0).to(device)
        loss_SmoothL1Loss_ev = torch.nn.SmoothL1Loss()(ev_embeds,ev_concept_pro.repeat(self.sample_num,1,1,1))
        loss_SmoothL1Loss_txt = torch.nn.SmoothL1Loss()(txt_embeds,txt_concept_pro.repeat(self.sample_num,1,1,1))

        # print(ev_embeds_.mean())
        # print(ev_embeds_)
        # print(txt_embeds)
        # print(ev_embeds_.shape)
        ev_embeds_ = ev_embeds.permute(0,2,1,3).reshape(num_concept*self.sample_num, B_ev, -1)
        txt_embeds_ = txt_embeds.permute(0,2,1,3).reshape(num_concept*self.sample_num, B_txt, -1)

        # [step5] Calculate Loss Function:
        # contrastive loss function
        if self.training:
            loss_cross_entroy = torch.tensor(0.0).to(device)
            for i in range(num_concept*self.sample_num):
                # the contrastive loss is only calculated for every concept.
                logits = logit_scale * ev_embeds_[i,:,:] @ txt_embeds_[i,:,:].t()
                loss_cross_entroy += symmetric_cross_entropy_loss(logits)
                # print(loss_cross_entroy)
                # print(logits)

            ev_f = ev_f.mean(1)
            logits_ori = logit_scale * ev_f @ txt_f.t()
            loss_cross_entroy_ori = symmetric_cross_entropy_loss(logits_ori)
            # print(loss_cross_entroy)
            # print(logits)

            loss_logsigma_sum, cov_ev_txt_loss, cov_ev_loss, cov_txt_loss = \
                torch.tensor(0.0).to(device), torch.tensor(0.0).to(device), torch.tensor(0.0).to(device), torch.tensor(0.0).to(device)
            # logsima_sum
            logsigma_sum = torch.exp(ev_logsigma).sum() + torch.exp(txt_logsigma).sum()
            # # print(logsigma_sum)
            # # loss_logsigma_sum = torch.tensor(max(0, self.loss_sigma_threshold - logsigma_sum)).to(device)
            loss_logsigma_sum = logsigma_sum.to(device)

            # # cross-correlation matrix
            # ev_embeds_norm = (ev_embeds - ev_embeds.mean(0)) / ev_embeds.std(0) # sample_num, B_ev, num_concept, concept_dim
            # ev_embeds_norm = ev_embeds_norm.mean(dim=0).reshape(num_concept*B_ev, -1) # B_ev*num_concept, concept_dim
            # txt_embeds_norm = (txt_embeds - txt_embeds.mean(0)) / txt_embeds.std(0) # sample_num, B_txt, num_concept, concept_dim
            # txt_embeds_norm = txt_embeds_norm.mean(dim=0).reshape(num_concept*B_txt, -1) # B_txt*num_concept, concept_dim
            # cov_ev_txt = torch.einsum('ab, bc->ac', [ev_embeds_norm, txt_embeds_norm.t()])
            # cov_ev = torch.einsum('ab, bc->ac', [ev_embeds_norm, ev_embeds_norm.t()])
            # cov_txt = torch.einsum('ab, bc->ac', [txt_embeds_norm, txt_embeds_norm.t()])
            # cov_ev_txt_loss = (cov_ev_txt.sum()-2*torch.diagonal(cov_ev_txt).sum())
            # cov_ev_loss = (cov_ev.sum()-2*torch.diagonal(cov_ev).sum())
            # cov_txt_loss = (cov_txt.sum()-2*torch.diagonal(cov_txt).sum())
            # print(cov_ev_txt_loss)
            # print(cov_ev_loss)
            # print(cov_txt_loss)

            # return loss_cross_entroy, loss_logsigma_sum, cov_ev_txt_loss, cov_ev_loss, cov_txt_loss
            return loss_cross_entroy, loss_cross_entroy_ori, loss_logsigma_sum,\
                loss_SmoothL1Loss_ev, loss_SmoothL1Loss_txt, \
                cov_ev_txt_loss, cov_ev_loss, cov_txt_loss

        else:

            # the last concept for Progressive Progress,
            # that represents the highest semantic meaning, is used for evaluation.
            logits = logit_scale * ev_embeds_[-1, :, :] @ txt_embeds_[-1, :, :].t()  # B_ev*B_txt

            ev_f = ev_f.mean(1)
            logits_ori = logit_scale * ev_f @ txt_f.t()  # B_ev*B_txt
            return logits, logits_ori

#+Concept Extraction+Event Concept fusion+Progressive Progress
class Conceptualizing_Alignment(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.num_concept = cfg['MODEL']['Conceptualizing_Alignment']['num_concept']
        self.concept_dim = cfg['MODEL']['Conceptualizing_Alignment']['concept_dim']
        self.sample_num = cfg['MODEL']['Conceptualizing_Alignment']['sample_num']
        self.loss_sigma_threshold = cfg['LossFunction']['loss_sigma_threshold']
        self.txt_Concept = Concept_Project_function(512, cfg)
        self.ev_Concept = Concept_Project_function(512, cfg)
        self.txt_Concept_fusion = nn.ModuleList()
        self.ev_Concept_fusion = nn.ModuleList()
        for i in range(self.num_concept // 2):
            self.txt_Concept_fusion.append(Concept_Fusion_function(self.num_concept*self.concept_dim//pow(2,i), cfg))
            self.ev_Concept_fusion.append(Concept_Fusion_function(self.num_concept*self.concept_dim//pow(2,i), cfg))
        self.ev_gau_encoder = DisTrans(self.concept_dim, 8)
        self.txt_gau_encoder = DisTrans(self.concept_dim, 8)
        if cfg['Trainer']['Precision'] == "fp16":
            self.ev_gau_encoder.half()
            self.txt_gau_encoder.half()

    def Content_Uncertainty(self, ev_concept, txt_concept):

        # [step3] Content Uncertainty:
        ev_mu, ev_logsigma, _ = self.ev_gau_encoder(ev_concept) #B, num_concept, concept_dim
        txt_mu, txt_logsigma, _ = self.txt_gau_encoder(txt_concept) #B, num_concept, concept_dim

        return ev_mu, ev_logsigma, txt_mu, txt_logsigma

    def Point_Feature_Sampling(self, ev_mu, ev_logsigma, txt_mu, txt_logsigma):
        # [step4] Point Feature Sampling:
        # ev = [ev_mu] * self.sample_num
        # txt = [txt_mu] * self.sample_num
        ev = []
        txt = []
        for i in range(self.sample_num):
            eps = torch.randn(ev_mu.shape[0], ev_mu.shape[1], ev_mu.shape[2], device=ev_mu.device)
            ev_i = ev_mu + torch.exp(ev_logsigma) * eps # B, num_concept, concept_dim
            ev.append(ev_i)

            eps = torch.randn(txt_mu.shape[0], txt_mu.shape[1], txt_mu.shape[2], device=txt_mu.device)
            txt_i = txt_mu + torch.exp(txt_logsigma) * eps # B, num_concept, concept_dim
            txt.append(txt_i)

        ev_embeds = torch.stack(ev) #self.sample_num B, num_concept, concept_dim
        txt_embeds = torch.stack(txt) #self.sample_num B, num_concept, concept_dim

        return ev_embeds, txt_embeds

    def forward(self, ev_f, txt_f, actual_event_length, logit_scale):
        """
        ev_f: extracted event feature, B,T,dim
        txt_f: extracted text feature, B,dim
        """
        device = txt_f.device
        # print(device)

        # # [step1] Concept Extraction:
        B_txt, txt_dim = txt_f.shape
        txt_concept = self.txt_Concept(txt_f).reshape(B_txt, self.num_concept, self.concept_dim).to(device) # B_txt, num_concept, concept_dim
        #
        B_ev, T, ev_dim = ev_f.shape
        ev_concept = []
        for i in range(T):
            ev_f_i = ev_f[:,i,:] # B, ev_dim
            ev_f_i = self.ev_Concept(ev_f_i).reshape(B_ev, self.num_concept, self.concept_dim) # B_ev, T, num_concept, concept_dim
            ev_concept.append(ev_f_i)
        ev_concept = torch.stack(ev_concept).to(device) # T, B_ev, num_concept, concept_dim
        # ev_concept = ev_concept.sum(0)

        # [step2] Event Concept fusion: fusion event frames by text concepts
        T_weight = torch.einsum('abde,cde->abc', [ev_concept, txt_concept]) # T, B_ev, B_txt
        T_weight = torch.mean(torch.mean(T_weight, dim=2), dim=1) # T
        for i in range(B_ev):
            T_weight[actual_event_length[i]:] = 0.0 # exclude the padded event frames
        T_weight = torch.softmax(T_weight, dim=0)  # T
        ev_concept = torch.einsum('abcd,a->bcd', [ev_concept, T_weight]).to(device) # B_ev, num_concept, concept_dim

        # [step6] Progressive Progress:
        # txt_concept_flatten = txt_concept.reshape(B_txt, self.num_concept*self.concept_dim)
        # ev_concept_flatten = ev_concept.reshape(B_ev, self.num_concept*self.concept_dim)
        # ev_concept_pro = [ev_concept]
        # txt_concept_pro = [txt_concept]
        # num_concept = self.num_concept
        # for i in range(self.num_concept // 2 - 1):
        #     txt_concept_i = self.txt_Concept_fusion[i](txt_concept_flatten).reshape(B_txt, self.num_concept//pow(2,i+1), self.concept_dim)
        #     ev_concept_i = self.ev_Concept_fusion[i](ev_concept_flatten).reshape(B_ev, self.num_concept//pow(2,i+1), self.concept_dim)
        #     ev_concept_pro.append(ev_concept_i)
        #     txt_concept_pro.append(txt_concept_i)
        #     txt_concept_flatten = txt_concept_i.reshape(B_txt, self.num_concept*self.concept_dim//pow(2,i+1))
        #     ev_concept_flatten = ev_concept_i.reshape(B_ev, self.num_concept*self.concept_dim//pow(2,i+1))
        #     num_concept += self.num_concept//pow(2,i+1)

        # ev_concept_pro = torch.cat(ev_concept_pro, dim=1).to(device)# B_ev, num_concept, concept_dim
        # txt_concept_pro = torch.cat(txt_concept_pro, dim=1).to(device)# B_txt, num_concept, concept_dim

        # [step3] Content Uncertainty:
        # ev_concept_pro = torch.mean(ev_f, dim=1).reshape(num_concept, B_ev, -1).permute(1,0,2) # B_ev, num_concept, concept_dim
        # txt_concept_pro = txt_f.reshape(num_concept, B_txt, -1).permute(1,0,2) # B_txt, num_concept, concept_dim

        ev_mu, ev_logsigma, txt_mu, txt_logsigma = self.Content_Uncertainty(ev_concept, txt_concept) # B_txt / B_ev, num_concept, concept_dim
        # [step4] Point Feature Sampling:
        ev_embeds, txt_embeds = \
            self.Point_Feature_Sampling(ev_mu, ev_logsigma, txt_mu, txt_logsigma)  # sample_num, B, num_concept, concept_dim
        ev_embeds_ = ev_embeds.permute(0,2,1,3).reshape(self.sample_num, self.num_concept*B_ev, -1).to(device)
        txt_embeds_ = txt_embeds.permute(0,2,1,3).reshape(self.sample_num, self.num_concept*B_txt, -1).to(device)

        # [step5] Calculate Loss Function:
        # contrastive loss function
        if self.training:

            return ev_embeds_, txt_embeds_, ev_f.mean(1), txt_f, ev_logsigma, txt_logsigma, \
                ev_concept, txt_concept, logit_scale.to(device)

        else:
            return ev_embeds_, txt_embeds_, ev_f.mean(1), txt_f, logit_scale.to(device)

if __name__ == '__main__':
    # test code
    cfg = read_yaml('F:\code\HumanECLIP\Configs/backbone_ddp.yaml')
    ev_f = torch.randn((5, 15, 512))
    txt_f = torch.randn((5, 512))
    logit_scale = 1.0
    actual_event_length = [3,1,5,3,5]
    model = Conceptualizing_Alignment(cfg).float()
    loss_cross_entroy, loss_logsigma_sum, cov_ev_txt_loss, cov_ev_loss, cov_txt_loss =\
        model(ev_f, txt_f, actual_event_length, logit_scale)
    print(loss_cross_entroy)
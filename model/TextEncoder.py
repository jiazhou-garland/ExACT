from collections import OrderedDict
import torch
import torch.nn as nn
from clip import clip
import json,os
import numpy as np

class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, SepcialPrompts, Original_tokenized_prompts):
        x = SepcialPrompts + self.positional_embedding  # (n_cls, n_tkn, ctx_dim)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        # Note that prompts learner utilize eot token as textual embedding(a.k.a. Original_tokenized_prompts.argmax(dim=-1))
        x = x[torch.arange(x.shape[0]), Original_tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x

class SpecificTextualPrompt(nn.Module):
    def __init__(self, cfg, clip_model, init_ctx = False, leranable_ctx=False):
        super().__init__()
        self.n_ctx = cfg['MODEL']['TextEncoder']['N_CTX']  # the number of textual prompts
        self.ctx_init = cfg['MODEL']['TextEncoder']['CTX_INIT']
        tf = open(cfg['Dataset']['Classnames'], "r")
        classnames_dict = json.load(tf)  # class name idx start from 0
        self.classnames_list = [i for i in classnames_dict.keys()]
        self.val_dic = {int(val): key for key, val in classnames_dict.items()}
        # print(self.val_dic)
        self.clip_model = clip_model
        self.dtype = clip_model.dtype
        self.ctx_dim = clip_model.ln_final.weight.shape[0]
        self.vis_dim = clip_model.visual.output_dim

        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg['Dataset']['Input_size'][0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if init_ctx:
            # use given words to initialize context vectors
            ctx_init = self.ctx_init.replace("_", " ")
            self.n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = self.clip_model.token_embedding(prompt).type(self.dtype)
            ctx_vectors = embedding[0, 1: 1 + self.n_ctx, :] # extract ctx without SOS and EOS
            self.prompt_prefix = ctx_init

        if leranable_ctx:
            # random context vectors initialization
            ctx_vectors = torch.empty(self.n_ctx, self.ctx_dim, dtype=self.dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            self.prompt_prefix = " ".join(["X"] * self.n_ctx)

        print(f'Initial context: "{self.prompt_prefix}"')
        print(f"Number of context words (tokens): {self.n_ctx}")
        self.ctx = nn.Parameter(ctx_vectors)

        # TODO: change the MetaNet here
        self.meta_net = nn.Sequential(OrderedDict([
            ("linear1", nn.Linear(self.vis_dim, self.vis_dim // 16)),
            ("relu", nn.ReLU(inplace=True)),
            ("linear2", nn.Linear(self.vis_dim // 16, self.ctx_dim))
        ]))
        if cfg['Trainer']['Precision'] == "fp16":
            self.meta_net.half()  # float32->loat16

    def construct_prompts(self, ctx, prefix, suffix, label=None):
        # dim0 is either batch_size (during training) or n_cls (during testing)
        # ctx: context tokens, with shape of (dim0, n_ctx, ctx_dim)
        # prefix: the sos token, with shape of (n_cls, 1, ctx_dim)
        # suffix: remaining tokens, with shape of (n_cls, *, ctx_dim)

        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]

        prompts = torch.cat(
            [
                prefix,  # (dim0, 1, dim) # SOS(start of special token)
                ctx,  # (dim0, n_ctx, dim) # textual prompts e.g.: a event frame of
                suffix,  # (dim0, *, dim) # after ctx_init, including name+. +EOS(end of special token)
            ],
            dim=1, )

        return prompts

    def forward(self, im_features, class_idxs, use_bias):
        device = im_features.device
        classnames = [self.val_dic[int(class_idxs[i])] for i in range(len(class_idxs))]
        # print(classnames)
        n_cls = len(classnames)
        # classnames = [name.replace("_", " ") for name in classnames]
        # name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [self.prompt_prefix + " " + name + "." for name in classnames]  # init_token + class name + .

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(device)  # (n_cls, n_text_token)
        with torch.no_grad():
            embedding = self.clip_model.token_embedding(tokenized_prompts).type(self.dtype).to(device)

        prefix = embedding[:, :1, :]
        suffix = embedding[:, 1 + self.n_ctx:, :]
        # text prompts e.g.: a event frame of
        ctx = self.ctx.unsqueeze(0).to(device)  # (1, n_ctx, ctx_dim)

        if use_bias == True:
            # add bias only on textual prompts
            bias = self.meta_net(im_features).to(device)  # (batch, ctx_dim)
            bias = bias.unsqueeze(1)  # (batch, 1, ctx_dim)
            ctx_shifted = ctx + bias  # (batch, n_ctx, ctx_dim)
        else:
            b, _, _ = im_features.size()
            _, n_ctx, ctx_dim = ctx.size()
            ctx_shifted = ctx.expand(b, n_ctx, ctx_dim)

        # Use instance-conditioned context tokens for all classes
        prompts = []
        for ctx_shifted_i in ctx_shifted:
            ctx_i = ctx_shifted_i.unsqueeze(0).expand(n_cls, -1, -1)
            pts_i = self.construct_prompts(ctx_i, prefix, suffix)  # (n_cls, n_tkn, ctx_dim)
            prompts.append(pts_i)
        SepcialPrompts = torch.stack(prompts).to(device)# (batch, n_cls, n_tkn, ctx_dim)

        return SepcialPrompts, tokenized_prompts

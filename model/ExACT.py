import torch
import torch.nn as nn
from .utils.clip_utils import load_clip_to_cpu
from .utils.yaml import read_yaml
from .TextEncoder import SpecificTextualPrompt, TextEncoder
from .EventEncoder import EventEncoder
from .Conceptualizing_Alignment import Conceptualizing_Alignment
from .ImageEncoder import ImageEncoder

class ECLIP(nn.Module):
    def __init__(self, cfg, clip_model_im, clip_model_ev):
        super().__init__()
        self.use_init_ctx = cfg['MODEL']['TextEncoder']['init_ctx']
        self.use_leranable_ctx = cfg['MODEL']['TextEncoder']['leranable_ctx']
        if self.use_init_ctx:
            self.prompt_learner_init = SpecificTextualPrompt(cfg, clip_model_im,
                                   init_ctx = cfg['MODEL']['TextEncoder']['init_ctx'])
        if self.use_leranable_ctx:
            self.prompt_learner_lernable = SpecificTextualPrompt(cfg, clip_model_im,
                                        leranable_ctx = cfg['MODEL']['TextEncoder']['leranable_ctx'])
        self.text_encoder = TextEncoder(clip_model_im)
        self.event_encoder = EventEncoder(cfg, clip_model_ev)
        # Todo:
        # Note that the image_encoder is not part of the ExAct model architecture,
        # however, I neglected to annotate this line during training so it must be initializazed when loading pre-trained models.
        self.image_encoder = ImageEncoder(cfg, clip_model_im)
        self.logit_scale = clip_model_im.logit_scale.exp()
        self.dtype = clip_model_im.dtype
        self.use_event_bias_textual_prompts = cfg['MODEL']['TextEncoder']['use_event_bias_textual_prompts']

    def forward(self, events, actual_event_length, class_idxs):
        event_features = self.event_encoder(events,actual_event_length) # b,T 512
        event_features = event_features / event_features.norm(dim=-1, keepdim=True)  # b,T,dim_e [2,512]

        if self.use_init_ctx:
            Text_Prompts_events_init, tokenized_prompts_ev_init = self.prompt_learner_init(event_features, class_idxs,
                                                                            self.use_event_bias_textual_prompts)

            Text_Prompts_events_init = Text_Prompts_events_init[0,:,:,:] # n_cls, n_tkn, ctx_dim
            text_features_e_init = self.text_encoder(Text_Prompts_events_init, tokenized_prompts_ev_init)
            text_features_e_init = text_features_e_init / text_features_e_init.norm(dim=-1, keepdim=True)  # b,dim_t_e [2,512]


        if self.use_leranable_ctx:
            Text_Prompts_events_lernable, tokenized_prompts_ev_lernable = self.prompt_learner_lernable(event_features, class_idxs,
                                                                            self.use_event_bias_textual_prompts)
            Text_Prompts_events_lernable = Text_Prompts_events_lernable[0,:,:,:] # n_cls, n_tkn, ctx_dim
            text_features_e_lernable = self.text_encoder(Text_Prompts_events_lernable, tokenized_prompts_ev_lernable)
            text_features_e_lernable = text_features_e_lernable / text_features_e_lernable.norm(dim=-1, keepdim=True)  # b,dim_t_e [2,512]

        if self.use_leranable_ctx and self.use_init_ctx:
            text_features = (text_features_e_init + text_features_e_lernable) / 2.0

        elif self.use_leranable_ctx:
            text_features = text_features_e_lernable

        else:
            text_features = text_features_e_init

        return event_features, text_features, self.logit_scale

class ExACT(nn.Module):

    def __init__(self, ECLIP, cfg):
        super().__init__()
        self.Conceptualizing_Alignment = Conceptualizing_Alignment(cfg)
        self.ECLIP = ECLIP

    def forward(self, events, actual_event_length, class_idxs):
        if self.training:
            event_features, text_features, logit_scale = self.ECLIP(events, actual_event_length, class_idxs)
            ev_embeds_, txt_embeds_, ev_f, txt_f, ev_logsigma, txt_logsigma,\
                ev_concept, txt_concept, logit_scale\
                = self.Conceptualizing_Alignment(event_features, text_features, actual_event_length, logit_scale)
            ev_embeds_ = ev_embeds_.reshape(ev_embeds_.shape[0]*ev_embeds_.shape[1], ev_embeds_.shape[2])
            txt_embeds_ = txt_embeds_.reshape(txt_embeds_.shape[0]*txt_embeds_.shape[1], txt_embeds_.shape[2])
            ev_concept = ev_concept.squeeze(1).repeat(5, 1, 1).reshape(ev_embeds_.shape[0], ev_embeds_.shape[1])
            txt_concept = txt_concept.squeeze(1).repeat(5, 1, 1).reshape(txt_embeds_.shape[0], txt_embeds_.shape[1])
            return ev_embeds_, txt_embeds_, ev_f, txt_f, ev_logsigma, txt_logsigma,\
                 ev_concept, txt_concept, logit_scale
        else:
            event_features, text_features, logit_scale = self.ECLIP(events, actual_event_length, class_idxs)
            ev_embeds_, txt_embeds_, ev_f, txt_f, logit_scale = self.Conceptualizing_Alignment(event_features, text_features, actual_event_length, logit_scale)
            ev_embeds_ = ev_embeds_.sum(0)
            txt_embeds_ = txt_embeds_.sum(0)
            return ev_embeds_, txt_embeds_, ev_f, txt_f, logit_scale

# Test code
if __name__ == '__main__':
    cfg = read_yaml('/ExACT_github\Configs/backbone_ddp.yaml')
    clip_model = load_clip_to_cpu(cfg).float()
    ECLIP = ECLIP(cfg, clip_model, clip_model).float()
    ExACT = ExACT(ECLIP, cfg).float()
    events = torch.randn((5, 3, 5, 224, 224))
    class_idxs = [0,1,2,3,4]
    actual_event_length = [3,1,5,3,5]
    # for name, param in model.named_parameters():
    #     print(name)
    loss_cross_entroy, loss_logsigma_sum, cov_loss = ExACT(events, actual_event_length, class_idxs)
    print(loss_cross_entroy)

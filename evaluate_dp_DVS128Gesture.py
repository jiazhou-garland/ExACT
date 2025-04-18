import torch
from torch.utils.data import DataLoader
import json, os
import numpy as np
from os.path import join, abspath, dirname
from model.utils.yaml import read_yaml
from model.utils.seed_torch import seed_torch
from model.ExACT import load_clip_to_cpu, ExACT, ECLIP
from Dataloader.DVS128Gesture_sampled_dataset import DVS128Gesture_sampled, event_sampled_frames_collate_func
import torch.nn as nn

def evaluate_one_epoch(model, cfg, dataloader):
    classnames_num = 11
    classnames_idxs = torch.from_numpy(np.arange(0, classnames_num))
    total, hit1_ev = 0, 0
    model = model.eval().float()
#------------------------------------------------------ev -> text----------------------------------------------
    for events, actual_event_length, labels in dataloader:
        # data, labels, = data.cuda(), labels.cuda()
        if cfg['MODEL']['BACKBONE']['PRE_ENCODING'] == "fp16":
            events = torch.from_numpy(events).float()
            # labels = torch.from_numpy(labels)
        with torch.no_grad():
            ev_embeds_, txt_embeds_, _, _, logit_scale = model(events, actual_event_length, classnames_idxs)
            if logit_scale.ndim != 0:
                logit_scale = logit_scale[0]
            logits = logit_scale * ev_embeds_ @ txt_embeds_.t()
            scores_ev = logits.softmax(dim=-1) # b,n

        B, _ = scores_ev.size()
        for i in range(B):
            total += 1
            scores_ev_i = scores_ev[i]
            label_i = labels[i]

            if classnames_idxs[scores_ev_i.topk(1)[1].cpu().detach().numpy()[0]] == label_i:
                hit1_ev += 1

            acc1_ev = hit1_ev / total * 100.


            if total % cfg['Trainer']['print_freq'] == 0:
                print(f'[Evaluation] num_samples: {total}  '
                      f'cumulative_acc1_ev: {acc1_ev:.2f}%  '
                      )
    print(f'Accuracy on validation set: ev_top1={acc1_ev:.2f}%.')

#     torch.cuda.empty_cache()
#     gc.collect()
    return acc1_ev, acc5_ev, acc1_ev_ori, acc5_ev_ori


if __name__ == '__main__':
    # ---------------------------------------------------init----------------------------------------------------------------
    THIS_DIR = abspath(dirname(__file__))
    cfg = read_yaml(THIS_DIR + '/Configs/DVS128Gesture.yaml')
    seed_torch(cfg['Trainer']['seed'])

    # -----------------------------------------------dataset----------------------------------------------------------------
    val_dataset = DVS128Gesture_sampled(cfg['Dataset']['Val']['Path'], cfg['Dataset']['Classnames'])
    val_loader = DataLoader(val_dataset, batch_size=cfg['Dataset']['Val']['Batch_size'],
                            shuffle=False, drop_last=True, num_workers=16, prefetch_factor=2, pin_memory=True,
                            collate_fn=event_sampled_frames_collate_func)

    # -------------------------------------------------model-----------------------------------------------------------------
    gpus = cfg['Trainer']['GPU_ids']
    device = torch.device("cuda:{}".format(gpus[0]) if torch.cuda.is_available() else "cpu")
    # print(device)
    print(f"Loading CLIP (backbone: {cfg['MODEL']['BACKBONE']['Name']})")
    clip_model_im = load_clip_to_cpu(cfg)
    for name, param in clip_model_im.named_parameters():
        param.requires_grad = False
        # print('Turn off the requires_grad of ' + name + " in clip_model_im into false. ")
    clip_model_ev = load_clip_to_cpu(cfg)
    for name, param in clip_model_ev.named_parameters():
        param.requires_grad = False
        # print('Turn on the requires_grad of ' + name + " in clip_model_ev into ture. ")

    print('----------------------------------------------------')
    print('Trainable Parameters')
    ECLIP = ECLIP(cfg, clip_model_im, clip_model_ev).to(device)
    for name, param in ECLIP.named_parameters():
        if "prompt_learner.clip_model" in name:
            param.requires_grad = False
        if "image_encoder" in name:
            param.requires_grad = False

    ExACT = ExACT(ECLIP, cfg).to(device)
    ExACT = nn.DataParallel(ExACT, device_ids=gpus, output_device=gpus[0])

    if cfg['MODEL']['Load_Path'] != None:
        ExACT.load_state_dict(torch.load(cfg['MODEL']['Load_Path'],map_location=torch.device('cpu')))
    #  for name, param in ExACT_original.named_parameters():
    #     if param.requires_grad == True:
    #         print('Trainable layers includes: ' + name)
    # print(ExACT_original)

    # ----------------------------------------------val----------------------------------------------------------------
    # torch.autograd.set_detect_anomaly(True)
    num_epochs = cfg['Trainer']['epoch']
    logit_scale = clip_model_im.logit_scale.exp()

    acc1_ev = evaluate_one_epoch(ExACT, cfg, val_loader)


import torch
from torch.utils.data import DataLoader
import gc, time, json, os, wandb
from os.path import join, abspath, dirname
import numpy as np
from model.utils.yaml import read_yaml
from model.utils.seed_torch import seed_torch
from model.ExACT import load_clip_to_cpu, ExACT, ECLIP
from Dataloader.DVS128Gesture_sampled_dataset import DVS128Gesture_sampled, event_sampled_frames_collate_func
from model.LossFunction import symmetric_cross_entropy_loss
import torch.nn as nn

def train_one_epoch(model, cfg, scaler, optimizer, scheduler, dataloader, epoch):
    epoch_start = time.time()
    length = len(dataloader)
    running_loss, dataset_size, loss, epoch_loss = 0.0, 0.0, 0.0, 0.0
    batch_size = cfg['Dataset']['Train']['Batch_size']
    num_concept = cfg['MODEL']['Conceptualizing_Alignment']['num_concept']
    sample_num = cfg['MODEL']['Conceptualizing_Alignment']['sample_num']
    for step, (events, actual_event_length, class_ids) in enumerate(dataloader):

        batch_start = time.time()
        model = model.train().float()
        if cfg['MODEL']['BACKBONE']['PRE_ENCODING'] == "fp16":
            events = torch.from_numpy(events).half()
            actual_event_length = torch.from_numpy(actual_event_length)
            class_ids = torch.from_numpy(class_ids)

        with torch.cuda.amp.autocast(enabled=True):
            ev_embeds_, txt_embeds_, ev_f, txt_f, ev_logsigma, txt_logsigma, \
                ev_concept, txt_concept, logit_scale = \
                model(events, actual_event_length, class_ids)
            if logit_scale.ndim != 0:
                logit_scale = logit_scale[0]

            device = ev_embeds_.device
            train_loss, loss_cross_entroy, loss_cross_entroy_ori, loss_logsigma_sum,\
                loss_SmoothL1Loss_ev, loss_SmoothL1Loss_txt = \
                torch.tensor(0.0).to(device), torch.tensor(0.0).to(device), torch.tensor(0.0).to(device),\
                torch.tensor(0.0).to(device), torch.tensor(0.0).to(device), torch.tensor(0.0).to(device)

            if cfg['LossFunction']['loss_cross_entroy']:
                logits = logit_scale * ev_embeds_ @ txt_embeds_.t()
                loss_cross_entroy = symmetric_cross_entropy_loss(logits)
                train_loss = train_loss + loss_cross_entroy
            if cfg['LossFunction']['loss_cross_entroy_ori']:
                logits_ori = logit_scale * ev_f @ txt_f.t()
                loss_cross_entroy_ori = symmetric_cross_entropy_loss(logits_ori)
                train_loss = train_loss + loss_cross_entroy_ori
            if cfg['LossFunction']['loss_logsigma_sum']:
                loss_logsigma_sum = torch.exp(ev_logsigma).sum() + torch.exp(txt_logsigma).sum()
                train_loss = train_loss + loss_logsigma_sum
            if cfg['LossFunction']['loss_SmoothL1Loss_ev']:
                loss_SmoothL1Loss_ev = torch.nn.SmoothL1Loss()(ev_embeds_, ev_concept)
                train_loss = train_loss + loss_SmoothL1Loss_ev
            if cfg['LossFunction']['loss_SmoothL1Loss_txt']:
                loss_SmoothL1Loss_txt = torch.nn.SmoothL1Loss()(txt_embeds_,txt_concept)
                train_loss = train_loss + loss_SmoothL1Loss_txt

            loss_list = torch.stack([train_loss.sum(),
                                     loss_cross_entroy.sum(), loss_cross_entroy_ori.sum(),
                                     loss_logsigma_sum.sum(), loss_SmoothL1Loss_ev.sum(), loss_SmoothL1Loss_txt.sum()], dim=0) \
                        / cfg['Trainer']['accumulation_steps']
        scaler.scale(loss_list[0]).backward()

        if (step + 1) % cfg['Trainer']['accumulation_steps'] == 0:
            scaler.step(optimizer)
            scaler.update()
            # zero the parameter gradients
            optimizer.zero_grad()

        running_loss += (loss_list.cpu().detach().numpy() * batch_size)
        dataset_size += batch_size
        epoch_loss = running_loss / dataset_size
        batch_end = time.time()

        if (step) % cfg['Trainer']['print_freq'] == 0:
            print(
                f'[{step + 1} / {length} | epoch: {epoch}] epoch_total_loss: {epoch_loss[0]:.7f} | '
                f'loss_cross_entroy: {epoch_loss[1]:.7f} | '
                f'loss_cross_entroy_ori: {epoch_loss[2]:.7f} | '
                f'loss_logsigma_sum: {epoch_loss[3]:.7f} | '
                f'loss_SmoothL1Loss_ev: {epoch_loss[4]:.7f} | '
                f'loss_SmoothL1Loss_txt: {epoch_loss[5]:.7f} | '
                f'lr: {optimizer.param_groups[0]["lr"]:.7f} | '
                f'batch_time: {(batch_end - batch_start):.3f} | '
                # f'memory_used: {torch.cuda.max_memory_allocated() / (1024.0 * 1024.0):.0f}MB'
            )
        if (step) % 500 == 0:
            wandb.log({'loss_cross_entroy': epoch_loss[1],
                       'loss_cross_entroy_ori': epoch_loss[2],
                       'loss_logsigma_sum': epoch_loss[3],
                       'loss_SmoothL1Loss_ev': epoch_loss[4],
                       'loss_SmoothL1Loss_txt': epoch_loss[5],
                       "LR": optimizer.param_groups[0]["lr"]})

    scheduler.step()
    epoch_time = time.time() - epoch_start
    print(f"EPOCH {epoch} training takes {epoch_time}s.")
    # torch.cuda.empty_cache()
    gc.collect()
    return epoch_loss

def evaluate_one_epoch(model, cfg, dataloader, classnames_num, logit_scale, evaluate_one_epoch=True):
    classnames_idxs = torch.from_numpy(np.arange(0, classnames_num))
    total, hit1_ev, hit5_ev, hit1_ev_ori, hit5_ev_ori = 0, 0, 0, 0, 0
    model = model.eval().float()
    all_logits_te_e, all_logits_te_e_ori = [], []
    all_label = []
#------------------------------------------------------ev -> text----------------------------------------------
    for events, actual_event_length, labels in dataloader:
        if cfg['MODEL']['BACKBONE']['PRE_ENCODING'] == "fp16":
            events = torch.from_numpy(events).float()
        with torch.no_grad():
            ev_embeds_, txt_embeds_, ev_f, txt_f, logit_scale = model(events, actual_event_length, classnames_idxs)

            if logit_scale.ndim != 0:
                logit_scale = logit_scale[0]
            txt_embeds_, txt_f = txt_embeds_[:11,:], txt_f[:11,:]
            logits = logit_scale * ev_embeds_ @ txt_embeds_.t()
            scores_ev = logits.softmax(dim=-1) # b,n
            logits_ori = logit_scale * ev_f @ txt_f.t()
            scores_ev_ori = logits_ori.softmax(dim=-1) # b,n
            all_logits_te_e.append(logits)
            all_logits_te_e_ori.append(logits_ori)

        B, _ = scores_ev.size()
        for i in range(B):
            total += 1
            scores_ev_i = scores_ev[i]

            label_i = labels[i]
            all_label.append(label_i)
            if scores_ev_i.topk(1)[1].cpu().detach().numpy()[0] == label_i:
                hit1_ev += 1
            if int(label_i) in set(scores_ev_i.topk(5)[1].cpu().detach().numpy()):
                hit5_ev += 1

            acc1_ev = hit1_ev / total * 100.
            acc5_ev = hit5_ev / total * 100.

            scores_ev_ori_i = scores_ev_ori[i]

            if scores_ev_ori_i.topk(1)[1].cpu().detach().numpy()[0] == label_i:
                hit1_ev_ori += 1
            if int(label_i) in set(scores_ev_ori_i.topk(5)[1].cpu().detach().numpy()):
                hit5_ev_ori += 1

            acc1_ev_ori = hit1_ev_ori / total * 100.
            acc5_ev_ori = hit5_ev_ori / total * 100.

            if total % cfg['Trainer']['print_freq'] == 0:
                print(f'[Evaluation] num_samples: {total}  '
                      f'cumulative_acc1_ev: {acc1_ev:.2f}%  '
                      f'cumulative_acc5_ev: {acc5_ev:.2f}%  '
                      f'cumulative_acc1_ev_ori: {acc1_ev_ori:.2f}%  '
                      f'cumulative_acc5_ev_ori: {acc5_ev_ori:.2f}%  '
                      )
    print(f'Accuracy on validation set: ev_top1={acc1_ev:.2f}%, ev_top5={acc5_ev:.2f}% '
          f'ev_ori_top1={acc1_ev_ori:.2f}%, ev_ori_top5={acc5_ev_ori:.2f}%')

    acc_retrival_1_e, acc_retrival_5_e, acc_retrival_10_e = 0,0,0
    acc_retrival_1_e_ori, acc_retrival_5_e_ori, acc_retrival_10_e_ori = 0,0,0

    return acc1_ev, acc5_ev, acc1_ev_ori, acc5_ev_ori,\
        acc_retrival_1_e, acc_retrival_5_e, acc_retrival_10_e,\
        acc_retrival_1_e_ori, acc_retrival_5_e_ori, acc_retrival_10_e_ori

def train_Conceptualizing_Alignment(ExACT, cfg):
    cfg['LossFunction']['loss_cross_entroy'] = True
    cfg['LossFunction']['loss_cross_entroy_ori'] = False
    cfg['LossFunction']['loss_logsigma_sum'] = True
    cfg['LossFunction']['loss_SmoothL1Loss_ev'] = True
    cfg['LossFunction']['loss_SmoothL1Loss_txt'] = True
    print('----------------------------------------------------')
    for name, param in ExACT.named_parameters():
        if "event_encoder" in name:
            param.requires_grad = False
        if "prompt_learner_lernable" in name:
            param.requires_grad = False
        if "prompt_learner_init" in name:
            param.requires_grad = False
        if "Conceptualizing_Alignment" in name:
            param.requires_grad = True
        if param.requires_grad == True:
            print('Trainable layers includes: ' + name)
    return ExACT, cfg

def train_EventEncoder(ExACT, cfg):
    cfg['LossFunction']['loss_cross_entroy'] = False
    cfg['LossFunction']['loss_cross_entroy_ori'] = True
    cfg['LossFunction']['loss_logsigma_sum'] = False
    cfg['LossFunction']['loss_SmoothL1Loss_ev'] = False
    cfg['LossFunction']['loss_SmoothL1Loss_txt'] = False
    print('----------------------------------------------------')
    # Train event encoder， frozen the Conceptualizing Alignment part
    for name, param in ExACT.named_parameters():
        if "Conceptualizing_Alignment" in name:
            param.requires_grad = False
        if "event_encoder" in name:
            param.requires_grad = True
        if "prompt_learner_lernable" in name:
            param.requires_grad = True
        if "prompt_learner_init" in name:
            param.requires_grad = True
        if "prompt_learner_init.clip_model" in name:
            param.requires_grad = False
        if param.requires_grad == True:
            print('Trainable layers includes: ' + name)
    return ExACT, cfg

def train_Conceptualizing_Alignment_EventEncoder(ExACT, cfg):
    cfg['LossFunction']['loss_cross_entroy'] = True
    cfg['LossFunction']['loss_cross_entroy_ori'] = False
    cfg['LossFunction']['loss_logsigma_sum'] = True
    cfg['LossFunction']['loss_SmoothL1Loss_ev'] = False
    cfg['LossFunction']['loss_SmoothL1Loss_txt'] = False
    for name, param in ExACT.named_parameters():
        if "Conceptualizing_Alignment" in name:
            param.requires_grad = True
        if "event_encoder" in name:
            param.requires_grad = True
        if "prompt_learner_lernable" in name:
            param.requires_grad = True
        if "prompt_learner_init" in name:
            param.requires_grad = True
        if "prompt_learner_init.clip_model" in name:
            param.requires_grad = False
        if param.requires_grad == True:
            print('Trainable layers includes: ' + name)
    return ExACT, cfg

if __name__ == '__main__':
    # ---------------------------------------------------init----------------------------------------------------------------
    cfg = read_yaml('./Configs/DVS128Gesture.yaml')
    THIS_DIR = abspath(dirname(__file__))
    RESULT_DIR = join(THIS_DIR, "Result")
    if not os.path.exists(RESULT_DIR):
        os.makedirs(RESULT_DIR)
    EXP_DIR = join(RESULT_DIR, f"{cfg['Trainer']['exp_group_name']}-" + str(cfg['Trainer']['exp_num']))
    if not os.path.exists(EXP_DIR):
        os.makedirs(EXP_DIR)

    run = wandb.init(project='ExACT_original',
                     config=cfg,
                     entity='garland-chou',
                     name=str(cfg['Trainer']['exp_num']),
                     group=cfg['Trainer']['exp_group_name'],
                     )
    seed_torch(cfg['Trainer']['seed'])
    tf = open(cfg['Dataset']['Classnames'], "r")
    classnames_dict = json.load(tf)  # class name idx start from 0
    classnames_list = [i for i in classnames_dict.keys()]
    classnames_num = len(classnames_list)
    # -----------------------------------------------dataset-----------------------------------------------------------------
    train_dataset = DVS128Gesture_sampled(cfg['Dataset']['Train']['Path'], cfg['Dataset']['Classnames'])
    train_loader = DataLoader(train_dataset, batch_size=cfg['Dataset']['Train']['Batch_size'],
                              shuffle=True, drop_last=True, num_workers=2, prefetch_factor=2, pin_memory=True,
                              collate_fn=event_sampled_frames_collate_func)

    val_dataset = DVS128Gesture_sampled(cfg['Dataset']['Val']['Path'], cfg['Dataset']['Classnames'])
    val_loader = DataLoader(val_dataset, batch_size=cfg['Dataset']['Val']['Batch_size'],
                            shuffle=True, drop_last=True, num_workers=2, prefetch_factor=2, pin_memory=True,
                            collate_fn=event_sampled_frames_collate_func)

    # -------------------------------------------------model-----------------------------------------------------------------
    gpus = cfg['Trainer']['GPU_ids']
    device = torch.device("cuda:{}".format(gpus[0]) if torch.cuda.is_available() else "cpu")
    print(device)
    print(f"Loading CLIP (backbone: {cfg['MODEL']['BACKBONE']['Name']})")
    clip_model_im = load_clip_to_cpu(cfg)
    for name, param in clip_model_im.named_parameters():
        param.requires_grad = False
    clip_model_ev = load_clip_to_cpu(cfg)
    for name, param in clip_model_ev.named_parameters():
        param.requires_grad = False

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
    # ----------------------------------------------optimizer&lr-------------------------------------------------------------
    optimizer = torch.optim.AdamW(ExACT.parameters(), lr=float(cfg['Trainer']['lr']),
                                  weight_decay=float(cfg['Trainer']['weight_decay']))
    lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=3*cfg['Trainer']['epoch'],
                                                          eta_min=float(cfg['Trainer']['min_lr']))
    loss_scaler = torch.cuda.amp.grad_scaler.GradScaler(enabled=True)

    # ----------------------------------------------train&val----------------------------------------------------------------
    num_epochs = cfg['Trainer']['epoch']
    logit_scale = clip_model_im.logit_scale.exp()

    ExACT, cfg = train_EventEncoder(ExACT, cfg)
    best_acc1_ev, best_acc5_ev = -np.inf, -np.inf
    for epoch in range(1, num_epochs + 1):
        epoch_loss = train_one_epoch(ExACT, cfg, loss_scaler, optimizer, lr_sched, train_loader, epoch)

        wandb.log({"train_epoch_total_loss": epoch_loss[0],
                   "loss_cross_entroy": epoch_loss[1],
                   "loss_cross_entroy_ori": epoch_loss[2],
                   "loss_logsigma_sum": epoch_loss[3],
                   "loss_SmoothL1Loss_ev": epoch_loss[4],
                   "loss_SmoothL1Loss_txt": epoch_loss[5]})

        acc1_ev, acc5_ev, acc1_ev_ori, acc5_ev_ori, \
            acc_retrival_1_e, acc_retrival_5_e, acc_retrival_10_e, \
            acc_retrival_1_e_ori, acc_retrival_5_e_ori, acc_retrival_10_e_ori \
            = evaluate_one_epoch(ExACT, cfg, val_loader, classnames_num, logit_scale, False)

        # Log the metrics
        wandb.log({"acc1_ev": acc1_ev, "acc5_ev": acc5_ev,
                   "acc1_ev_ori": acc1_ev_ori, "acc5_ev_ori": acc5_ev_ori,
                   'acc_retrival_1_e': acc_retrival_1_e,
                   'acc_retrival_5_e': acc_retrival_5_e,
                   'acc_retrival_10_e': acc_retrival_10_e,
                   'acc_retrival_1_e_ori': acc_retrival_1_e_ori,
                   'acc_retrival_5_e_ori': acc_retrival_5_e_ori,
                   'acc_retrival_10_e_ori': acc_retrival_10_e_ori,
                   "LR": lr_sched.get_last_lr()[0]})

    ExACT, cfg = train_Conceptualizing_Alignment(ExACT, cfg)
    best_acc1_ev, best_acc5_ev = -np.inf, -np.inf
    for epoch in range(1, num_epochs + 1):
        epoch_loss = train_one_epoch(ExACT, cfg, loss_scaler, optimizer, lr_sched, train_loader, epoch)

        wandb.log({"train_epoch_total_loss": epoch_loss[0],
                   "loss_cross_entroy": epoch_loss[1],
                   "loss_cross_entroy_ori": epoch_loss[2],
                   "loss_logsigma_sum": epoch_loss[3],
                   "loss_SmoothL1Loss_ev": epoch_loss[4],
                   "loss_SmoothL1Loss_txt": epoch_loss[5]})

        acc1_ev, acc5_ev, acc1_ev_ori, acc5_ev_ori, \
            acc_retrival_1_e, acc_retrival_5_e, acc_retrival_10_e, \
            acc_retrival_1_e_ori, acc_retrival_5_e_ori, acc_retrival_10_e_ori \
            = evaluate_one_epoch(ExACT, cfg, val_loader, classnames_num, logit_scale, False)

        # Log the metrics
        wandb.log({"acc1_ev": acc1_ev, "acc5_ev": acc5_ev,
                   "acc1_ev_ori": acc1_ev_ori, "acc5_ev_ori": acc5_ev_ori,
                   'acc_retrival_1_e': acc_retrival_1_e,
                   'acc_retrival_5_e': acc_retrival_5_e,
                   'acc_retrival_10_e': acc_retrival_10_e,
                   'acc_retrival_1_e_ori': acc_retrival_1_e_ori,
                   'acc_retrival_5_e_ori': acc_retrival_5_e_ori,
                   'acc_retrival_10_e_ori': acc_retrival_10_e_ori,
                   "LR": lr_sched.get_last_lr()[0]})


    ExACT, cfg = train_Conceptualizing_Alignment_EventEncoder(ExACT, cfg)
    best_acc1_ev, best_acc5_ev = -np.inf, -np.inf
    for epoch in range(1, num_epochs + 1):
        epoch_loss = train_one_epoch(ExACT, cfg, loss_scaler, optimizer, lr_sched, train_loader, epoch)

        wandb.log({"train_epoch_total_loss": epoch_loss[0],
                   "loss_cross_entroy": epoch_loss[1],
                   "loss_cross_entroy_ori": epoch_loss[2],
                   "loss_logsigma_sum": epoch_loss[3],
                   "loss_SmoothL1Loss_ev": epoch_loss[4],
                   "loss_SmoothL1Loss_txt": epoch_loss[5]})

        acc1_ev, acc5_ev, acc1_ev_ori, acc5_ev_ori, \
            acc_retrival_1_e, acc_retrival_5_e, acc_retrival_10_e, \
            acc_retrival_1_e_ori, acc_retrival_5_e_ori, acc_retrival_10_e_ori \
            = evaluate_one_epoch(ExACT, cfg, val_loader, classnames_num, logit_scale, False)

        # Log the metrics
        wandb.log({"acc1_ev": acc1_ev, "acc5_ev": acc5_ev,
                   "acc1_ev_ori": acc1_ev_ori, "acc5_ev_ori": acc5_ev_ori,
                   'acc_retrival_1_e': acc_retrival_1_e,
                   'acc_retrival_5_e': acc_retrival_5_e,
                   'acc_retrival_10_e': acc_retrival_10_e,
                   'acc_retrival_1_e_ori': acc_retrival_1_e_ori,
                   'acc_retrival_5_e_ori': acc_retrival_5_e_ori,
                   'acc_retrival_10_e_ori': acc_retrival_10_e_ori,
                   "LR": lr_sched.get_last_lr()[0]})

        # save model based on event
        if acc1_ev >= best_acc1_ev:
            print(f"acc Improved ({best_acc1_ev:0.4f}% ---> {acc1_ev:0.4f}%), ({best_acc5_ev:0.4f}% ---> {acc5_ev:0.4f}%)")
            best_acc1_ev, best_acc5_ev = acc1_ev, acc5_ev
            PATH = join(EXP_DIR, f"train_Conceptualizing_Alignment_EventEncoder_best_ev_epoch.bin")
            torch.save(ExACT.state_dict(), PATH)
            print(f"Model Saved at {PATH}")
        print(f"the best acc1_ev is {best_acc1_ev}%.")
        wandb.log({"bset_acc1_ev": best_acc1_ev, "best_acc5_ev": best_acc5_ev})


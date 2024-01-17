# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Train and eval functions used in main.py
"""
import math
import sys
from typing import Iterable, Optional

import torch

from timm.data import Mixup
from timm.utils import accuracy, ModelEma

#from losses import DistillationLoss
import lib.test.analysis.utils

import numpy as np
#sys.path.remove("/home/sun/anaconda3/envs/DEKR/lib/python3.8/site-packages")
import cv2
#import matplotlib.pyplot as plt

from PIL import Image
#from calc_flops import calc_flops, throughput, calc_flops_images


# def train_one_epoch(model: torch.nn.Module, criterion: DistillationLoss,
#                     data_loader: Iterable, optimizer: torch.optim.Optimizer,
#                     device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
#                     model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
#                     set_training_mode=True, args = None):
#     model.train(set_training_mode)
#     metric_logger = utils.MetricLogger(delimiter="  ")
#     metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
#     header = 'Epoch: [{}]'.format(epoch)
#     print_freq = 100

#     l1_lambda = 0.001

#     for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
#         samples = samples.to(device, non_blocking=True)
#         targets = targets.to(device, non_blocking=True)

#         if mixup_fn is not None:
#             samples, targets = mixup_fn(samples, targets)

#         if args.bce_loss:
#             targets = targets.gt(0.0).type(targets.dtype)

#         with torch.cuda.amp.autocast():
#             outputs, weights, attention_mask = model(samples)
#             loss = criterion(samples, outputs, targets)

#             B = outputs.shape[0]

#             l1_penalty = 0.
#             for layer in attention_mask:
#                 l1_penalty += torch.norm(layer,1)
#             l1_penalty /= len(attention_mask)
#             l1_penalty *= l1_lambda
#             print(l1_penalty)

#             loss = loss # + l1_penalty

#         loss_value = loss.item()

#         if not math.isfinite(loss_value):
#             print("Loss is {}, stopping training".format(loss_value))
#             sys.exit(1)

#         optimizer.zero_grad()

#         # this attribute is added by timm on one optimizer (adahessian)
#         is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
#         loss_scaler(loss, optimizer, clip_grad=max_norm,
#                     parameters=model.parameters(), create_graph=is_second_order)

#         torch.cuda.synchronize()
#         if model_ema is not None:
#             model_ema.update(model)

#         metric_logger.update(loss=loss_value)
#         metric_logger.update(lr=optimizer.param_groups[0]["lr"])
#     # gather the stats from all processes
#     metric_logger.synchronize_between_processes()
#     print("Averaged stats:", metric_logger)
#     return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

# # FLOPs: 4.608338304
# @torch.no_grad()
# def evaluate(data_loader, model, device):
#     criterion = torch.nn.CrossEntropyLoss()

#     metric_logger = utils.MetricLogger(delimiter="  ")
#     header = 'Test:'

#     # switch to evaluation mode
#     model.eval()
#     l1_lambda = 0.001

#     flops_list = []
#     cnt = 0

#     for images, target in metric_logger.log_every(data_loader, 5000, header):
#         images = images.to(device, non_blocking=True)
#         target = target.to(device, non_blocking=True)

#         cnt += 1

#         with torch.cuda.amp.autocast():
#             if utils.is_main_process():
#                 flops = calc_flops_images(model, images)
#                 print('FLOPs: {}'.format(flops))
#             try:
#                 output, weights, attention_mask = model(images)
#             except:
#                 continue
#             loss = criterion(output, target)

#             # l1_penalty = 0.
#             # for layer in attention_mask:
#             #     l1_penalty += torch.norm(layer,1)
#             # l1_penalty /= len(attention_mask)
#             # l1_penalty *= l1_lambda
#             #print(l1_penalty)

#             loss = loss # + l1_penalty

#         acc1, acc5 = accuracy(output, target, topk=(1, 5))
#         #
#         batch_size = images.shape[0]
#         metric_logger.update(loss=loss.item())
#         metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
#         metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
#         metric_logger.meters['flops'].update(flops, n=batch_size)
#         # if cnt < 25000:
#         #     metric_logger.meters['flops'].update(flops, n=batch_size)

#     # gather the stats from all processes
#     metric_logger.synchronize_between_processes()
#     # print("mean FLOPS : ", mean(flops_list))
#     print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f} flops {losses.global_avg:.3f}'
#           .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss)) # , flops=metric_logger.flops
#     # print('* Acc@1 {top1.global_avg:.3f}'
#           # .format(flops=metric_logger.flops))


#     return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
def get_attention_map(img, att_mat):  
    att_mat = torch.stack(att_mat).squeeze(0)

    # att_mat = torch.mean(torch.mean(att_mat, dim=2), dim=1)  # dim=1)
    att_mat = torch.mean(torch.max(att_mat, dim=1)[0], dim=1)
    
    att_mat = (att_mat/torch.max(att_mat)).detach().cpu()

    v = att_mat[0]
    grid_size = int(np.sqrt(att_mat.size(-1)))

    mask = v.reshape(grid_size, grid_size).detach().numpy()

    mask = cv2.resize(mask / mask.max(), (img.shape[0], img.shape[1]))[..., np.newaxis]
    mask = cv2.applyColorMap((mask*255.).astype("uint8"), cv2.COLORMAP_JET)
    result = (mask*0.4 + img*0.6).astype("uint8")
    return result

def get_attention_map2(img, att_mat, swin_att_mat=None, get_fusion=False, get_dino=False, get_decoder=False, swin_attn=False, get_mask=False):  
    if not get_fusion:
        att_mat = torch.stack(att_mat).squeeze(0)
    # print("att_mat.shape : ", att_mat.shape)
    
    if swin_att_mat is not None:
        swin_att_mat = torch.stack(swin_att_mat).squeeze(1)
        # print("att_mat.shape : ", swin_att_mat.shape)

    # Average the attention weights across all heads.
    if not swin_attn and not get_fusion and not get_decoder:
        att_mat = torch.mean(att_mat, dim=1).detach().cpu()
    elif get_decoder:
        # att_mat = torch.mean(torch.mean(att_mat, dim=1), dim=1).detach().cpu() # torch.mean(att_mat, dim=1).detach().cpu()
        att_mat = torch.mean(torch.max(att_mat, dim=1)[0], dim=1).detach().cpu()
        # att_mat = (att_mat/torch.max(att_mat)).detach().cpu()
    elif not get_fusion and not get_decoder and not get_dino:
        # att_mats = torch.mean(att_mat, dim=1).squeeze(0).detach().cpu()
        att_mat = torch.mean(torch.mean(att_mat, dim=1), dim=1).detach().cpu()
        # att_mat = torch.max(torch.max(att_mat, dim=2)[0], dim=1)[0].detach().cpu()
    
    # result_list = []
    # for att_mat in att_mats:
    #     att_mat = att_mat.unsqueeze(0)
    if swin_att_mat is not None:
        swin_att_mat = torch.mean(swin_att_mat, dim=1).detach().cpu()

    # # cls_token_swint - Edit by Minho 2023.02.05
    # swint_cls = torch.mean(att_mat, dim=1).unsqueeze(1)
    # att_mat = torch.cat((swint_cls, att_mat), 1)

    # swint_cls2 = torch.mean(att_mat, dim=1).unsqueeze(1)

    # To account for residual connections, we add an identity matrix to the
    # attention matrix and re-normalize the weights.
    if not get_fusion and not swin_attn:
        # residual_att = torch.eye(att_mat.size(1))  # [50, 50]
        # aug_att_mat = att_mat + residual_att  # [1, 50, 50]
        # aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)  # [1, 50, 50]

        # # Recursively multiply the weight matrices
        # joint_attentions = torch.zeros(aug_att_mat.size())  # [1, 50, 50]
        # joint_attentions[0] = aug_att_mat[0]

        # for n in range(1, aug_att_mat.size(0)):
        #     joint_attentions[n] = torch.matmul(aug_att_mat[n], joint_attentions[n-1])

        # v = joint_attentions[-1]
        v = att_mat[0]
        grid_size = int(np.sqrt(att_mat.size(-1)))
    if get_fusion or swin_attn:
        residual_att = torch.eye(att_mat.size(1))  # [49, 49]
        aug_att_mat = att_mat.unsqueeze(1) + residual_att
        aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)

        # Recursively multiply the weight matrices
        joint_attentions = torch.zeros(aug_att_mat.size())
        joint_attentions[0] = aug_att_mat[0]

        for n in range(1, aug_att_mat.size(0)):
            joint_attentions[n] = torch.matmul(aug_att_mat[n], joint_attentions[n-1])

        v = joint_attentions[-1]
        grid_size = int(np.sqrt(att_mat.size(-1)))
    
    if swin_att_mat is not None:
        swin_residual_att = torch.eye(swin_att_mat.size(1))
        swin_aug_att_mat = swin_att_mat + swin_residual_att
        swin_aug_att_mat = swin_aug_att_mat / swin_aug_att_mat.sum(dim=-1).unsqueeze(-1)

        # Recursively multiply the weight matrices
        swin_joint_attentions = torch.zeros(swin_aug_att_mat.size())
        swin_joint_attentions[0] = swin_aug_att_mat[0]

        for n in range(1, swin_aug_att_mat.size(0)):
            swin_joint_attentions[n] = torch.matmul(swin_aug_att_mat[n], swin_joint_attentions[n-1])

        swin_v = swin_joint_attentions[-1]
        swin_grid_size = int(np.sqrt(swin_aug_att_mat.size(-1)))
    
    #v = torch.mean(v, dim=0)
    if get_dino:
        mask = v[0, 1:].reshape(grid_size, grid_size).detach().numpy()
    elif get_decoder:
        mask = v.reshape(grid_size, grid_size).detach().numpy()
    elif get_fusion:
        mask = att_mat.reshape(grid_size, grid_size).detach().cpu().numpy()
    else:
        if len(att_mat)==4:
            mask_list = []
            for i in att_mat:
                v = i.clone()
                mask = v.reshape(grid_size, grid_size).detach().numpy()
                mask_list.append(mask)
        else:
            v = torch.mean(att_mat, dim=0)
            mask = v.reshape(grid_size, grid_size).detach().numpy()
    if swin_att_mat is not None:
        swin_v = torch.mean(swin_v, dim=0)
        swin_mask = swin_v.reshape(swin_grid_size, swin_grid_size).detach().numpy()

    if get_mask:
        result = cv2.resize(mask / mask.max(), (112,112))
    elif swin_att_mat is not None:
        fusion_mask = mask * swin_mask
        mask = cv2.resize(fusion_mask / fusion_mask.max(), (img.shape[0], img.shape[1]))[..., np.newaxis]
        mask = cv2.applyColorMap((mask*255.).astype("uint8"), cv2.COLORMAP_JET)
        result = (mask*0.4 + img*0.6).astype("uint8")
        return result
    else:
        if len(att_mat)==4:
            for num, i in enumerate(mask_list):
                i_mask = i / i.max()
                mask = cv2.resize(i_mask, (int(img.shape[0]/2), int(img.shape[1]/2)))[..., np.newaxis]
                mask = cv2.applyColorMap((mask*255.).astype("uint8"), cv2.COLORMAP_JET)
                if num == 0:
                    img[:int(img.shape[0]/2), :int(img.shape[0]/2), :] = (mask*0.4 + img[:int(img.shape[0]/2),
                                        :int(img.shape[0]/2), :]*0.6).astype("uint8")
                elif num == 1:
                    img[:int(img.shape[0]/2), int(img.shape[0]/2):, :] = (mask*0.4 + img[:int(img.shape[0]/2),
                                        int(img.shape[0]/2):, :]*0.6).astype("uint8")
                elif num == 2:
                    img[int(img.shape[0]/2):, :int(img.shape[0]/2), :] = (mask*0.4 + img[int(img.shape[0]/2):,
                                        :int(img.shape[0]/2), :]*0.6).astype("uint8")
                elif num == 3:
                    img[int(img.shape[0]/2):, int(img.shape[0]/2):, :] = (mask*0.4 + img[int(img.shape[0]/2):,
                                        int(img.shape[0]/2):, :]*0.6).astype("uint8")
            return img
        else:
            mask = cv2.resize(mask / mask.max(), (img.shape[0], img.shape[1]))[..., np.newaxis]
            mask = cv2.applyColorMap((mask*255.).astype("uint8"), cv2.COLORMAP_JET)
            result = (mask*0.4 + img*0.6).astype("uint8")
            return result

    # return img  # result  # img # result

def plot_attention_map(original_img, att_map):
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(16, 16))
    ax1.set_title('Original')
    ax2.set_title('Attention Map Last Layer')
    _ = ax1.imshow(original_img)
    _ = ax2.imshow(att_map)

def plot_attention_map(original_img, att_map,cnt):
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(16, 16))
    ax1.set_title('Original')
    ax2.set_title('Layer 3')
    _ = ax1.imshow(original_img)
    _ = ax2.imshow(att_map)
    # plt.imsave(str(cnt) + "_origin.png", original_img)
    plt.imsave("output/CIFAR10/visualize_output/prune/" + str(cnt) + "_layer9.png", att_map)
    # plt.show()

def show_mask_on_image(img, mask):
    img = np.float32(img) / 255
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

@torch.no_grad()
def visualize(data_loader, model, device):
    # IMAGENET_MEAN, IMAGENET_STD = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
    IMAGENET_MEAN, IMAGENET_STD = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    cnt = 0

    for images, target in data_loader: #metric_logger.log_every(data_loader, 10, header):
        cnt+=1
        if cnt % 1000 != 0:
            continue
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output, weights, pruning_mask = model(images)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)



        B = images.shape[0]

        for idx in range(images.shape[0]):
            cv_image = images.detach().cpu().numpy()[idx]
            cv_image = np.transpose(cv_image, (1, 2, 0))
        # print("cv_image.shape",cv_image)

            cv_image = np.clip(255.0 * (cv_image * IMAGENET_STD + IMAGENET_MEAN), 0, 255).astype("uint8")
        #print("/cv_image.shape1",cv_image.shape)
        # print("output/CIFAR10/visualize_output/prune/cv_image.shape",cv_image)
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            # print(len(weights))
            # print(weights[idx].shape)
            attention_image = get_attention_map(cv_image, weights)#,get_mask=True)
            cv2.imshow("attention_image",attention_image)

        # cv2.imwrite("cv_image.png", cv_image)
            layer_3_pruning_mask = pruning_mask[0].detach().cpu().numpy()[idx].reshape(14,-1)
            layer_6_pruning_mask = pruning_mask[1].detach().cpu().numpy()[idx].reshape(14,-1)
            layer_9_pruning_mask = pruning_mask[2].detach().cpu().numpy()[idx].reshape(14,-1)

            # print(layer_3_pruning_mask.shape)
            # print(layer_6_pruning_mask.shape)
            # print(layer_9_pruning_mask.shape)

            layer_3_pruning_mask = cv2.resize(layer_3_pruning_mask.astype(np.uint8), (224,224), interpolation=cv2.INTER_NEAREST).reshape(224,224,1)
            layer_6_pruning_mask = cv2.resize(layer_6_pruning_mask.astype(np.uint8), (224,224), interpolation=cv2.INTER_NEAREST).reshape(224,224,1)
            layer_9_pruning_mask = cv2.resize(layer_9_pruning_mask.astype(np.uint8), (224,224), interpolation=cv2.INTER_NEAREST).reshape(224,224,1)
            cv2.imshow("cv_image_1", cv_image*layer_3_pruning_mask)
            cv2.imshow("cv_image_2", cv_image*layer_6_pruning_mask)
            cv2.imshow("cv_image_3", cv_image*layer_9_pruning_mask)
            cv2.waitKey(-1)
            if cnt > 100:
                cv2.imwrite("vis_collection/"+str((cnt-1)*B+idx)+"_original_result.png", cv_image)
                # cv2.imwrite("vis_collection/"+str((cnt-1)*B+idx)+"_attention_result.png", attention_image)
                cv2.imwrite("vis_collection/"+str((cnt-1)*B+idx)+"_layer_3_pruning_result.png", cv_image*layer_3_pruning_mask)
                cv2.imwrite("vis_collection/"+str((cnt-1)*B+idx)+"_layer_6_pruning_result.png", cv_image*layer_6_pruning_mask)
                cv2.imwrite("vis_collection/"+str((cnt-1)*B+idx)+"_layer_9_pruning_result.png", cv_image*layer_9_pruning_mask)

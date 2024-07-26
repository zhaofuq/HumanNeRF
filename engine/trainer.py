# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import logging

#from apex import amp
import torch
import torch.nn.functional as F

from utils import batchify_ray
from utils.metrics import *
import numpy as np
import os
import time

def do_train(
        cfg, 
        model,
        extractor,
        train_loader,
        dataset,
        dataset_val,
        optimizer,
        scheduler,
        loss_fn,
        swriter,
        resume_iter=0
):
    max_epochs = cfg.SOLVER.MAX_EPOCHS
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    output_dir = cfg.OUTPUT_DIR
    coarse_stage = cfg.SOLVER.COARSE_STAGE
    psnr_metrics = cfg.SOLVER.PSNR
    scale = cfg.SOLVER.SCALE
    finetune = cfg.DATASETS.FINETUNE

    logger = logging.getLogger("RFRender.%s.train" % cfg.OUTPUT_DIR.split('/')[-1])
    logger.info("Start training")
    #global step
    global_step = 0
    torch.autograd.set_detect_anomaly(True)

    for epoch in range(1,max_epochs):
        model.cuda()

        #psnr monitor 
        psnr_monitor = []

        #epoch time recording
        epoch_start = time.time()

        #load train data
        for batch_id, batch in enumerate(train_loader):

            #split batch 
            batch, mbatch = split_batch(batch, cfg.SOLVER.IMS_PER_BATCH)
          
            for mb in range(mbatch):
                
                iters_start = time.time()
                global_step = (epoch -1) * len(train_loader) * mbatch + batch_id * mbatch  + mb

                model.train()
                optimizer.zero_grad()
                
                rays = batch['rays'][mb].cuda() # [batch_size,7]
                rgbs = batch['rgbs'][mb].cuda() # [batch_size,3-6]
                bboxes = batch['bboxes'][mb].cuda() # [batch_size, 8, 3]
                near_fars = batch['near_fars'][mb].cuda() # [batch_size, 2]
                mask = batch['mask'][mb].cuda()
                imgs = batch['imgs'][mb].cuda()
                scene = {'scene': batch['scene'][mb],'Ks':batch['Ks'][mb].cuda(), 'Ts':batch['Ts'][mb].cuda()}
                
                if cfg.DATASETS.USE_SKELETON:
                    Rts = batch['Rts'][mb].squeeze(1).cuda() # [batch_size, 24, 16]
                    global_ts = batch['global_ts'][mb].squeeze(1).cuda() # [batch_size, 3]
                    skeletons = batch['skeletons'][mb].squeeze(1).cuda() # [batch_size, 24, 3] -> [batch_size, 24*T, 3]
                    vertices = batch['vertices'][mb].squeeze(1).cuda()  #[vertice_num, 3]
                else:
                    Rts = None
                    global_ts = None
                    skeletons = None
                    vertices = None

                if epoch < 10 and not finetune:
                    feature_maps = extractor(2 * imgs - 1.0)
                else:
                    with torch.no_grad():
                        feature_maps = extractor(2 * imgs - 1.0)

                if epoch < coarse_stage:
                    stage2, stage1, ray_mask = model(rays, bboxes, feature_maps, 
                                                    Rts = Rts, global_ts = global_ts, skeletons = skeletons, vertices = vertices, scene = scene, 
                                                    only_coarse = False,  near_far=near_fars, scale=scale)
                else:
                    stage2, stage1, ray_mask = model(rays, bboxes, feature_maps, 
                                                    Rts = Rts, global_ts = global_ts, skeletons = skeletons, vertices = vertices, scene = scene, 
                                                    only_coarse = False, near_far=near_fars, scale=scale)

                #-------------------------------------------regularization----------------------------------------------#

                #rgb regularization
                if ray_mask is not None:
                    loss1 = loss_fn(stage2[0][ray_mask], rgbs[ray_mask])
                    loss2 = loss_fn(stage1[0][ray_mask], rgbs[ray_mask])
                else:
                    loss1 = loss_fn(stage2[0], rgbs)
                    loss2 = loss_fn(stage1[0], rgbs)

                if epoch < coarse_stage:
                    loss = loss2      
                else:
                    loss = loss1 + loss2 

                #proxy empty space regularization
                if cfg.SOLVER.USE_EMPTY_LOSS:
                    lambda_empty = 0.1
                    lambda_rgb = 1.0
                    mask = (mask > 0.5).float()

                    mask_loss_fine = F.binary_cross_entropy(stage2[2].clip(1e-3, 1.0 - 1e-3), mask)
                    mask_loss_coarse = F.binary_cross_entropy(stage1[2].clip(1e-3, 1.0 - 1e-3), mask)

                    if epoch < coarse_stage:
                        loss = lambda_rgb * loss + lambda_empty * mask_loss_coarse      
                    else:

                        loss = lambda_rgb * loss + lambda_empty * (mask_loss_fine + mask_loss_coarse)

                # with amp.scale_loss(loss, optimizer) as scaled_loss:
                #     scaled_loss.backward()
                #-------------------------------------------regularization----------------------------------------------#

                loss.backward()
                optimizer.step()
            
                with torch.no_grad():
                    if ray_mask is not None:
                        psnr_ = psnr(stage2[0][ray_mask], rgbs[ray_mask])
                    else:
                        psnr_ = psnr(stage2[0], rgbs)

                    if global_step % 50 == 0: 
        
                        swriter.add_scalars('TrainLoss', {'loss':loss.item()}, global_step)
                        swriter.add_scalars('TrainPsnr', {'psnr':psnr_}, global_step)

                        swriter.add_scalars('TrainLoss', {'rgb_loss_fine':loss2.item()}, global_step)
                        swriter.add_scalars('TrainLoss', {'rgb_loss_coarse':loss1.item()}, global_step)
                        if cfg.SOLVER.USE_EMPTY_LOSS:
                            swriter.add_scalars('TrainLoss', {'mask_loss_fine':mask_loss_fine.item()}, global_step)
                            swriter.add_scalars('TrainLoss', {'mask_loss_coarse':mask_loss_coarse.item()}, global_step)

                    #log train info every log_period
                    if global_step % log_period == 0:
                        for param_group in optimizer.param_groups:
                            lr = param_group['lr']
                        logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3e}  Psnr: {:.2f} Lr: {:.2e} Speed: {:.1f}[rays/s]"
                                    .format(epoch,  batch_id, len(train_loader), loss.item(), psnr_ ,lr,
                                            log_period * float(cfg.SOLVER.IMS_PER_BATCH) / (time.time() - iters_start)))
                    #validation
                    if global_step % 1000 == 0:
                        val_vis(cfg, model, extractor, dataset_val, loss_fn, swriter, logger, epoch, global_step)

                    #model saving
                    if global_step % checkpoint_period == 0:
                        ModelCheckpoint(model, extractor, optimizer, scheduler, output_dir, epoch)
                
                #ITERATION COMPLETED
                scheduler.step()
            
        #EPOCH COMPLETED
        ModelCheckpoint(model, extractor, optimizer, scheduler, output_dir, epoch)

        val_vis(cfg, model, extractor, dataset_val, loss_fn, swriter, logger, epoch, global_step)

        logger.info('Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[rays/s]'
                    .format(epoch, time.time() - epoch_start,
                            len(train_loader) * float(cfg.SOLVER.IMS_PER_BATCH) / (time.time() - epoch_start)))

def val_vis(cfg, model, extractor, dataset_val, loss_fn, swriter, logger, epoch ,iters):
    
    avg_loss, avg_psnr ,val_frame, val_cam = evaluator(cfg, model, extractor, dataset_val, loss_fn, swriter, iters)
    logger.info("Validation Results - Epoch: {} Avg Loss: {:.3f} Avg Psnr: {:.2f}"
                .format(epoch, avg_loss, avg_psnr)
                )
    swriter.add_scalars('ValidationLoss', {'exp':avg_loss}, iters)
    swriter.add_scalars('ValidationPsnr', {'exp':avg_psnr}, iters)
    swriter.add_scalars('Validation', {'Frame':val_frame}, iters)
    swriter.add_scalars('Validation', {'Cam':val_cam}, iters)


def evaluator(cfg, model, extractor,  dataset_val, loss_fn, swriter, iters):
    model.eval()
    
    sample = dataset_val.__getitem__(0)

    rays = sample['rays'].cuda()
    rgbs = sample['rgbs'].cuda()
    bboxes = sample['bboxes'].cuda()
    color_gt = sample['color'].cuda()
    mask = sample['mask'].cuda()
    
    cam_ids = sample['cam_ids']
    frame_ids = sample['frame_ids']
    near_far = sample['near_fars'].cuda()

    imgs = sample['imgs'].cuda()
    scene = {'scene': sample['scene'], 'Ks':sample['Ks'].cuda(), 'Ts':sample['Ts'].cuda()}

    if cfg.DATASETS.USE_SKELETON:
        Rts = sample['Rts'].cuda()
        global_ts = sample['global_ts'].cuda()
        skeletons = sample['skeletons'].cuda()
        vertices = sample['vertices'].cuda() 
    else:
        Rts = None
        global_ts = None
        skeletons = None
        vertices = None

    with torch.no_grad():
        feature_maps = extractor(2 * imgs - 1.0)

        stage2, stage1, _ = batchify_ray(model, rays, bboxes, feature_maps, 
                                        Rts = Rts, global_ts = global_ts, skeletons = skeletons, vertices = vertices, scene = scene, 
                                        chuncks = cfg.SOLVER.IMS_PER_BATCH , near_far=near_far, scale = cfg.SOLVER.SCALE)

        color_1 = stage2[0]
        depth_1 = stage2[1]
        acc_map_1 = stage2[2]

        color_0 = stage1[0]
        depth_0 = stage1[1]
        acc_map_0 = stage1[2]

        #Mask Visulization
        if cfg.DATASETS.USE_MASK: 
            mask = mask.reshape((color_gt.size(1), color_gt.size(2), 1)).permute(2, 0, 1)
            swriter.add_image('GT/mask', mask, iters)
        
        # Features Visulization
        vis_feats = []
        for i in range(feature_maps.shape[0]):
            vis_feats.append(feature_maps[i,:3,:,:])
        
        vis_feats = torch.cat(vis_feats, dim = -1)
        swriter.add_image('GT/feats', vis_feats, iters)

        # Output Visulization
        color_img = color_1.reshape((color_gt.size(1), color_gt.size(2), 3)).permute(2, 0, 1)
        depth_img = depth_1.reshape((color_gt.size(1), color_gt.size(2), 1)).permute(2, 0, 1)
        depth_img = (depth_img - depth_img.min()) / (depth_img.max() - depth_img.min())
        acc_map = acc_map_1.reshape((color_gt.size(1), color_gt.size(2), 1)).permute(2, 0, 1)

        color_img_0 = color_0.reshape((color_gt.size(1), color_gt.size(2), 3)).permute(2, 0, 1)
        depth_img_0 = depth_0.reshape((color_gt.size(1), color_gt.size(2), 1)).permute(2, 0, 1)
        depth_img_0 = (depth_img_0 - depth_img_0.min()) / (depth_img_0.max() - depth_img_0.min())
        acc_map_0 = acc_map_0.reshape((color_gt.size(1), color_gt.size(2), 1)).permute(2, 0, 1)
 
        swriter.add_image('GT/rgb', color_gt, iters)
        
        swriter.add_image('stage2/rendered', color_img, iters)
        swriter.add_image('stage2/depth', depth_img, iters)
        swriter.add_image('stage2/alpha', acc_map, iters)
        
        swriter.add_image('stage1/rendered', color_img_0, iters)
        swriter.add_image('stage1/depth', depth_img_0, iters)
        swriter.add_image('stage1/alpha', acc_map_0, iters)

        return loss_fn(color_img, color_gt).item(), psnr(color_img, color_gt), frame_ids[0].long(), cam_ids[0].long()

def ModelCheckpoint(model, extractor, optimizer, scheduler, output_dir, epoch):
    # model, optimizer, scheduler saving 
    torch.save(model.state_dict(),os.path.join(output_dir, 'model_epoch_%d.pth'%epoch))
    torch.save(extractor.state_dict(), os.path.join(output_dir, 'extractor_epoch_%d.pth'%epoch))
    torch.save(optimizer.state_dict(), os.path.join(output_dir, 'optimizer_epoch_%d.pth'%epoch))
    torch.save(scheduler.state_dict(), os.path.join(output_dir, 'scheduler_epoch_%d.pth'%epoch))


def split_batch(batch, chuncks):

    batch['rays'] = batch['rays'].squeeze(0).split(chuncks, dim=0)
    batch['rgbs'] = batch['rgbs'].squeeze(0).split(chuncks, dim=0)
    batch['bboxes'] = batch['bboxes'].squeeze(0).split(chuncks, dim=0)
    batch['near_fars'] = batch['near_fars'].squeeze(0).split(chuncks, dim=0)

    if 'imgs' in batch.keys():
        batch['imgs'] = batch['imgs'].squeeze(0).split(chuncks, dim=0)

    if 'mask' in batch.keys():
        batch['mask'] = batch['mask'].squeeze(0).split(chuncks, dim=0)

    if 'depth' in batch.keys():
        batch['depth'] = batch['depth'].squeeze(0).split(chuncks, dim=0)

    if 'bgs' in batch.keys():
        batch['bgs'] = batch['bgs'].squeeze(0).split(chuncks, dim=0)

    if 'Rts' in batch.keys():
        batch['Rts'] = batch['Rts'].squeeze(0).split(chuncks, dim=0)

    if 'global_ts' in batch.keys():
        batch['global_ts'] = batch['global_ts'].squeeze(0).split(chuncks, dim=0)

    if 'skeletons' in batch.keys():
        batch['skeletons'] = batch['skeletons'].squeeze(0).split(chuncks, dim=0)
        
    return batch, len(batch['rays'])
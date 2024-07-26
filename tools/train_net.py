import os
import shutil
import sys
import glob

from apex import amp
sys.path.append('..')
from config import cfg
from data import make_data_loader, make_data_loader_view
from engine.trainer import do_train
from modeling import build_model, build_extractor
from solver import make_optimizer, build_scheduler
from layers import make_loss

from utils.logger import setup_logger

from torch.utils.tensorboard import SummaryWriter
import torch

if len(sys.argv) > 2:
    torch.cuda.set_device(int(sys.argv[2]))
    training_folder = sys.argv[1]
    assert os.path.exists(training_folder), 'training_folder does not exist.'
    cfg.merge_from_file(os.path.join(training_folder, 'configs.yml'))
    cfg.freeze()
    output_dir = cfg.OUTPUT_DIR
    writer = SummaryWriter(log_dir=output_dir)
else:
    torch.cuda.set_device(int(sys.argv[1]))
    cfg.merge_from_file('../configs/config.yml')
    cfg.DATASETS.MODE = 'full'
    cfg.freeze()
    output_dir = cfg.OUTPUT_DIR
    writer = SummaryWriter(log_dir=output_dir)
    shutil.copy('../configs/config.yml', os.path.join(output_dir, 'configs.yml'))

writer.add_text('OUT_PATH', output_dir, 0)
logger = setup_logger("RFRender", output_dir, 0)
logger.info("Running with config:\n{}".format(cfg))

#Dataloader 
train_loader, dataset = make_data_loader(cfg, is_train=True)
val_loader, dataset_val = make_data_loader_view(cfg, is_train=False)

#MODEL 
model = build_model(cfg).cuda()
extractor = build_extractor(cfg).cuda()
optimizer = make_optimizer(cfg, model)
scheduler = build_scheduler(optimizer, cfg.SOLVER.WARMUP_ITERS, cfg.SOLVER.START_ITERS, cfg.SOLVER.END_ITERS,
                            cfg.SOLVER.LR_SCALE)

#MODEL Initilization
if os.path.exists(cfg.MODEL.MODEL_PRETRAINED):
    logger.info('load model params from path: '+ cfg.MODEL.MODEL_PRETRAINED)

    model_path = glob.glob(os.path.join(cfg.MODEL.MODEL_PRETRAINED,'*.pth'))
    if len(model_path) > 0:
        model_iter = [int(pth.replace('.pth','').split('_')[-1]) for pth in model_path]
        epoch = max(model_iter)

        model.load_state_dict(torch.load(os.path.join(cfg.MODEL.MODEL_PRETRAINED, 'model_epoch_%d.pth'%epoch), map_location='cpu'))
        extractor.load_state_dict(torch.load(os.path.join(cfg.MODEL.MODEL_PRETRAINED, 'extractor_epoch_%d.pth'%epoch), map_location='cpu'))
        optimizer.load_state_dict(torch.load(os.path.join(cfg.MODEL.MODEL_PRETRAINED, 'optimizer_epoch_%d.pth'%epoch), map_location='cpu'))
        scheduler.load_state_dict(torch.load(os.path.join(cfg.MODEL.MODEL_PRETRAINED, 'scheduler_epoch_%d.pth'%epoch), map_location='cpu'))

if cfg.DATASETS.FINETUNE:
    logger.info('freeze params......')
    for k,v in model.named_parameters():
        if 'blendnet' in k:
            v.requires_grad = False

    for k,v in extractor.named_parameters():
        v.requires_grad = False

loss_fn = make_loss(cfg)

# model, optimizer = amp.initialize(model, optimizer, opt_level="O1")

do_train(
    cfg,
    model,
    extractor,
    train_loader,
    dataset,
    dataset_val,
    optimizer,
    scheduler,
    loss_fn,
    writer,
    resume_iter=0
)

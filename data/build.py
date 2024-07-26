# encoding: utf-8
"""
@author:  Minye Wu
@GITHUB: wuminye
"""

from torch.utils import data
from .datasets.ray_source import Dataset_NHR, Dataset_View , Dataset_NeRF, Dataset_NB, Dataset_NeRF_Batch, Dataset_MultiNeRF, Dataset_MultiView
from .transforms import build_transforms

def build_dataset(data_folder_path,  transforms, bunch, cfg):

    if cfg.DATASETS.DATASET_NAME =='NHR':
        datasets = Dataset_NHR(data_folder_path, transforms=transforms, bunch=bunch, cfg=cfg)
    elif cfg.DATASETS.DATASET_NAME =='NERF':
        datasets = Dataset_NeRF(data_folder_path, transforms=transforms, bunch=bunch, cfg=cfg)
    elif cfg.DATASETS.DATASET_NAME =='NERF_BATCH':
        datasets = Dataset_NeRF_Batch(data_folder_path, transforms=transforms, bunch=bunch, cfg=cfg)
    elif cfg.DATASETS.DATASET_NAME =='NB':
        datasets = Dataset_NB(data_folder_path, transforms=transforms, bunch=bunch, cfg=cfg)
    elif cfg.DATASETS.DATASET_NAME =='MultiNeRF':
        datasets = Dataset_MultiNeRF(data_folder_path, transforms=transforms, bunch=bunch, cfg=cfg)
    return datasets

def build_dataset_view(data_folder_path,  transforms, cfg):
    if cfg.DATASETS.DATASET_NAME =='MultiNeRF':
        datasets = Dataset_MultiView(data_folder_path, transforms=transforms, cfg = cfg)
    else:
        datasets = Dataset_View(data_folder_path, transforms=transforms, cfg = cfg)
    return datasets

def make_data_loader(cfg, is_train=True):

    batch_size = cfg.SOLVER.IMS_PER_BATCH
    
    if is_train:
        batch_size = cfg.SOLVER.IMS_PER_BATCH
        shuffle = True
    else:
        batch_size = cfg.TEST.IMS_PER_BATCH
        shuffle = False

    if cfg.DATASETS.SPLIT_RAYS:
        batch_size = 1

    transforms = build_transforms(cfg, is_train)
    num_workers = cfg.DATALOADER.NUM_WORKERS

    if cfg.DATASETS.DATASET_NAME =='NHR':
        datasets = build_dataset(cfg.DATASETS.TRAIN, transforms, cfg.SOLVER.IMS_PER_BATCH, cfg = cfg)
        data_loader = data.DataLoader(
            datasets, batch_size = batch_size, shuffle=shuffle, num_workers=num_workers
        )
    else:
        datasets = build_dataset(cfg.DATASETS.TRAIN, transforms, cfg.SOLVER.IMS_PER_BATCH, cfg = cfg)
        data_loader = data.DataLoader(
            datasets, batch_size = 1, shuffle=True, num_workers=num_workers
        )
        
    return data_loader, datasets


def make_data_loader_view(cfg, is_train=False):

    batch_size = cfg.TEST.IMS_PER_BATCH

    transforms = build_transforms(cfg, is_train)
    datasets = build_dataset_view(cfg.DATASETS.TRAIN, transforms, cfg)

    num_workers = cfg.DATALOADER.NUM_WORKERS
    data_loader = data.DataLoader(
        datasets, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )

    return data_loader, datasets

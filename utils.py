import segmentation_models_pytorch as smp
import argparse
from torch.utils.data import Dataset, DataLoader, sampler
import pandas as pd
import os
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import torchvision.transforms.functional as TFF
import torchvision.transforms as TF
from PIL import Image
import segmentation_models_pytorch as smp
import imgaug.augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
from tqdm.auto import tqdm

ATTRIBUTES = ['globules', 'milia_like_cyst', 'negative_network', 'pigment_network', 'streaks']

def create_model(name, num_classes):
    return smp.Unet(name, encoder_weights='imagenet', classes=num_classes)

def create_optimizer(model, lr):
    return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.001)

def read_image(pt, size, is_image=True):
    img = Image.open(pt)
    H,W, = np.array(img).shape[:2]
    img2 = TFF.center_crop(img, min(H,W))
    interp_mode = Image.BILINEAR if is_image else Image.NEAREST
    img3 = TFF.resize(img2, size, interpolation=interp_mode)
    return np.array(img3)

def get_train_tfm():
    return iaa.Sequential([
        iaa.Fliplr(p=0.5),
        iaa.Flipud(p=0.5),
        # TF.RandomResizedCrop(299, scale=(0.75, 1.0)),
        iaa.Rotate(rotate=(-45,45)),
        iaa.AddToBrightness(),
        iaa.AddToHueAndSaturation((-50, 50))
    ])

class Collector:
    def __init__(self):
        self.vals = []
    def put(self, vals):
        self.vals.append(vals.detach().cpu().numpy())
    def get(self):
        if len(self.vals[0].shape):
            return np.concatenate(self.vals, axis=0)
        else:
            return np.stack(self.vals, axis=0)
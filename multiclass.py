from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
from pathlib import Path
import torch
import torchvision.transforms as TF
import torch.nn as nn
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
from tqdm.auto import tqdm
from utils import *



class MulticlassDataset(Dataset):
    def __init__(self, fldr_pt, size, trn_tfm=None):
        img_fldr = fldr_pt / 'images'
        mask_fldr = fldr_pt / 'multiclass/masks'
        self.names = sorted(map(lambda x: x.name, img_fldr.iterdir()))
        self.classes = sorted(os.listdir(mask_fldr))
        self.images = [read_image(img_fldr / name, size, is_image=True) for name in tqdm(self.names, desc='Load images')]
        self.masks = [read_image(mask_fldr / name, size, is_image=False) for name in tqdm(self.names, desc=f'Load masks', leave=False)]
        self.trn_tfm = trn_tfm
        self.to_tensor = TF.Compose([TF.ToTensor(), TF.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        
    def __len__(self): return len(self.names)
    
    def __getitem__(self, i):
        image = self.images[i]
        masks = self.masks[i]
        if self.trn_tfm is not None:
            segmap = SegmentationMapsOnImage(masks, shape=image.shape)
            res = self.trn_tfm(image=image, segmentation_maps=segmap)
            image, masks = res[0], res[1].arr[:,:,0]
        masks = torch.tensor(masks).long()
        return self.to_tensor(image), masks

    def get_classnames(self):
        return np.unique(self.masks).tolist()

def get_binary(mask, class_idx): return (mask == class_idx).int()
def get_intersection(lbl, pred): return (lbl * pred).sum()
def get_union(lbl, pred): return ((lbl + pred) > 0).int().sum()


def main(args):
    trn_ds = MulticlassDataset(Path('data_test'), size=args.image_size, trn_tfm=get_train_tfm())
    val_ds = MulticlassDataset(Path('data_test'), size=args.image_size)
    assert trn_ds.get_classnames() == val_ds.get_classnames(), "class names mismatch"
    trn_dl = DataLoader(trn_ds, batch_size=args.batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    classes = val_ds.get_classnames()

    model = create_model(args.model, num_classes=len(classes)).cuda()
    optimizer = create_optimizer(model, args.lr)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(args.num_epochs):
        # train
        model.train()
        trn_losses = Collector()
        for img, lbl in trn_dl:
            optimizer.zero_grad()
            out = model(img.cuda())
            loss = loss_fn(out, lbl.cuda())
            loss.backward()
            optimizer.step()
            trn_losses.put(loss)

        # valid
        model.eval()
        intersections, unions, val_losses = Collector(), Collector(), Collector()
        with torch.no_grad():
            for img, lbl in val_dl:
                out = model(img.cuda())
                loss = loss_fn(out, lbl.cuda())
                val_losses.put(loss)

                out = out.detach().cpu()
                pred = out.argmax(dim=1)              
                intersection = [get_intersection(get_binary(lbl, i), get_binary(pred, i)) for i in range(len(classes))]
                union = [get_union(get_binary(lbl, i), get_binary(pred, i)) for i in range(len(classes))]
                intersection = torch.tensor(intersection).unsqueeze(0)
                union = torch.tensor(union).unsqueeze(0)
                intersections.put(intersection)
                unions.put(union)

        log = {'epoch': epoch, 'loss/trn': trn_losses.get().mean(), 'loss/val': val_losses.get().mean()}

        IoUs = intersections.get().sum(axis=0) / unions.get().sum(axis=0)
        # we skip 0th class since it represents background
        for m, cl in enumerate(classes[1:], start=1):
            log[f'IoU/{cl}'] = IoUs[m]
        log['IoU/macro'] = IoUs[1:].mean()
        print(log)
    return model
    
def get_args():
    args = type('', (), {})()
    args.batch_size = 2 # 64
    args.cuda_idx = 1 # on which cuda to run
    args.model='resnet18'
    args.lr = 1e-3
    args.num_epochs = 2
    args.image_size = 224
    return args

if __name__ == '__main__':
    args = get_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_idx)
    main(args)

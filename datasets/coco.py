# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
COCO dataset which returns image_id for evaluation.

Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
"""
from pathlib import Path

import rasterio
import os
import numpy as np
from PIL import Image


import torch
import torch.utils.data
import torchvision
from pycocotools import mask as coco_mask

import datasets.transforms as T
from util.box_ops import get_true_centroid

class CocoDetection(torch.utils.data.Dataset):
    """`MS Coco Captions <http://mscoco.org/dataset/#detections-challenge2016>`_ Dataset.

    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    def __init__(self, image_set, img_folder, ann_file, transforms, return_masks, box_scale):
        from pycocotools.coco import COCO
        self.img_folder = img_folder
        self.coco = COCO(ann_file)
        self.image_set = image_set

        self.box_scale = box_scale
        
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks)

        if image_set == 'val':
            self.ids = list(sorted(self._build_valid_ids()))
        else:
            self.ids = list(sorted(self.coco.imgs.keys()))

    def _build_valid_ids(self):
        valid_ids = []
        for img_id in self.coco.imgs.keys():
            ann_ids = self.coco.getAnnIds(img_id)
            target = self.coco.loadAnns(ann_ids)
            

            path = self.coco.loadImgs(img_id)[0]['file_name']

            with rasterio.open(os.path.join(self.img_folder, path)) as src:
                band = src.read()
                img = np.repeat(band,3,axis=0).transpose(1,2,0).astype('float32')
            target = {'image_id': img_id, 'annotations': target}
            img, target = self.prepare(img, target, self.box_scale)
            if self._transforms is not None:
                img, target = self._transforms(img, target)
            if len(target['boxes']) != 0:
                valid_ids.append(img_id)
        return valid_ids
        
    def __getitem__(self, idx):
        """
        Args:
            index (int): Index

        Returns:
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """
        coco = self.coco
        img_id = self.ids[idx]
        box_scale = self.box_scale

        ann_ids = coco.getAnnIds(img_id)
        target = coco.loadAnns(ann_ids)


        path = coco.loadImgs(img_id)[0]['file_name']

        with rasterio.open(os.path.join(self.img_folder, path)) as src:
            band = src.read()
            img = np.repeat(band,3,axis=0).transpose(1,2,0).astype('float32')

        target = {'image_id': img_id, 'annotations': target}
        img, target = self.prepare(img, target, box_scale)
        if self._transforms is not None:
            img, target = self._transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.ids)


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image, target, box_scale):
        w, h = image.shape[0], image.shape[1]
        
        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)
        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)
        
        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        return image, target


def make_coco_transforms(image_set, crop):

    normalize = T.Compose([
        #T.ToTensor(),
        T.Normalize([6.6374, 6.6374, 6.6374], [10.184, 10.184, 10.184])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if image_set == 'train':
        return T.Compose([
            T.ToTensor(),
            T.RandomCrop((crop, crop)),
            T.RandomHorizontalFlip(),
            normalize,
            # T.RandomSelect(
            #     T.RandomResize(scales, max_size=1333),
            #     T.Compose([
            #         T.RandomResize([400, 500, 600]),
            #         T.RandomSizeCrop(384, 600),
            #         T.RandomResize(scales, max_size=1333),
            #     ])
            # ),
        ])

    if image_set == 'val':
        return T.Compose([
            T.ToTensor(),
            T.CenterCrop((crop,crop)),
            normalize,
            # T.RandomResize([800], max_size=1333),
        ])

    raise ValueError(f'unknown {image_set}')


def build(image_set, args):
    root = Path(args.coco_path)
    assert root.exists(), f'provided COCO path {root} does not exist'
    PATHS = {
        "train": (root / "train", root / "annotations" / 'train.json'),
        "val": (root / "validate", root / "annotations" / 'validate.json'),
    }
    img_folder, ann_file = PATHS[image_set]
    dataset = CocoDetection(image_set, img_folder, ann_file, transforms=make_coco_transforms(image_set, args.crop), return_masks=args.masks, box_scale=args.box_scale)
    return dataset

from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
from torchvision import transforms
import torch
import pickle
import json
import numpy as np
from pathlib import Path


class VGDataset(Dataset):
    def __init__(self, data_file, image_dir, T, h, w, transform=None):
        with open(data_file, 'r') as f:
            dataset = json.load(f)

        self.image_dir = image_dir
        self.transform = transform
        self.id2vocab = dataset['id2vocab']
        self.vocab2id = dataset['vocab2id']
        self.images = dataset['images']
        self.nvocab = len(self.id2vocab)
        self.T = T
        self.h = h
        self.w = w

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        data = self.images[idx]

        image_id = data['image_id']
        image = Image.open(Path(self.image_dir, '{}.jpg'.format(image_id))).convert("RGB")
        if self.transform: image = self.transform(image)

        object_ids = torch.LongTensor(data['object_ids'])
        object_bboxes = torch.Tensor(data['object_bboxes'])

        T, h, w, nvocab = self.T, self.h, self.w, self.nvocab
        # object_bbox_fms = torch.zeros(T, nvocab, h, w)
        object_bbox_masks = torch.zeros(T, h, w)
        object_areas = torch.zeros(2, h, w)

        for t, (object_id, object_bbox) in enumerate(zip(object_ids, object_bboxes)):
            if object_id == 0: break

            x0_, y0_, x1_, y1_ = object_bboxes[t]

            center_x = torch.round((x1_ + x0_)/2*w).long()
            center_y = torch.round((y1_ + y0_)/2*h).long()
            height = torch.round((y1_ - y0_)*h).long()
            width = torch.round((x1_ - x0_)*w).long()

            assert 0 <= float(center_x) <= w
            assert 0 <= float(center_y) <= h

            object_bbox_masks[t, center_y, center_x] = 1
            # todo: consider over-ride of point later
            object_areas[0, center_y, center_x] = height
            object_areas[1, center_y, center_x] = width

        sample = {'image': image,
                  'object_ids': object_ids,
                  'object_bboxes': object_bboxes,
                  'object_bbox_masks': object_bbox_masks,
                  'object_areas': object_areas
                  }

        return sample


def get_loader(data_file, image_dir, T, h, w, batch_size):
    transform = transforms.Compose([
        transforms.Resize((h, w)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    dataset = VGDataset(data_file, image_dir, T, h, w, transform)

    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    return data_loader


def get_flatten_objects(object_ids, object_bboxes):
    device = object_ids.device
    object_ids_flat = []
    object_bboxes_flat = []
    object_to_image_flat = []

    batch_size = object_ids.shape[0]
    for i in range(batch_size):
        idx = (object_ids[i] > 0).nonzero().view(-1)

        for j in idx:
            object_ids_flat.append(object_ids[i, j])
            object_bboxes_flat.append(object_bboxes[i, j])
            object_to_image_flat.append(i)

    object_ids_flat = torch.LongTensor(object_ids_flat).to(device)
    object_bboxes_flat = torch.stack(object_bboxes_flat, 0).to(device)
    object_to_image_flat = torch.LongTensor(object_to_image_flat).to(device)

    return object_ids_flat, object_bboxes_flat, object_to_image_flat


if __name__ == '__main__':
    data_file = '/media/lily/HDD8T/scene_generation/datasets/VG1.4/data/dataset_vocab_109_objnum_1_10_imgnum_100382_val.json'
    image_dir = '/media/lily/HDD8T/scene_generation/datasets/VG1.4/all_images/VG_100K'

    batch_size = 2
    T, h, w = 10, 64, 64

    data_loader = get_loader(data_file, image_dir, T, h, w, batch_size)

    for i, data in enumerate(data_loader):
        images = data['image']
        object_ids = data['object_ids']
        object_bboxes = data['object_bboxes']
        object_bbox_masks = data['object_bbox_masks']
        object_areas = data['object_areas']

        mask_norm = torch.sum(object_bbox_masks, dim=1, keepdim=True) + 0.0000001

        object_bbox_masks_norm = object_bbox_masks / mask_norm

        test1 = object_bbox_masks.numpy()
        print(images.shape)
        print(object_ids)
        print(object_bboxes * 64)
        print(object_bbox_masks.shape)

        object_ids_flat, object_bboxes_flat, object_to_image_flat = get_flatten_objects(object_ids, object_bboxes)

        print(object_ids_flat)
        print(object_bboxes_flat)
        print(object_to_image_flat)
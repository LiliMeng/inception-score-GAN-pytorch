'''
Author: Lili Meng (menglili@cs.ubc.ca)
Date: Oct 17th, 2018
License: MIT
'''

from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
from PIL import Image
from torchvision import transforms



class GeneratedVGDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.dataset_frame = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataset_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.dataset_frame.loc[idx, 'img_name'])
        image = Image.open(img_name)
        

        if self.transform:
            image = self.transform(image)

        sample = {'image': image}

        return sample

def get_loader(csv_file, img_dir, image_size, batch_size, mode='val', dataset='vg'):
    """Build and return data loader."""

    if dataset == 'VG':
        
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        dataset = GeneratedVGDataset(csv_file, img_dir, transform)
    else:
    	raise Exception("currently only VG generated images dataset is provided ")

    shuffle = True if mode == 'train' else False
    
    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4)

    return data_loader


if __name__ == '__main__':


    csv_file = './data/VG_generated_imgs.csv'
    img_dir = '/media/lily/HDD8T/scene_generation/sg2im/output_batch_img64'
    data_loader = get_loader(csv_file, img_dir, image_size=64, batch_size=32, mode='val',
                             dataset='VG')

    for i, data in enumerate(data_loader):
        img_batch = data['image']
        

        print('image size: {}'.format(img_batch.size()))
       
        break
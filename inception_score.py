import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
import torch.utils.data

from torchvision.models.inception import inception_v3

import numpy as np
from scipy.stats import entropy
from org_img_dataloader import *
from generated_img_dataloader import *

def inception_score(dataset, N, cuda=True, batch_size=32, resize=False, splits=1):
    """Computes the inception score of the generated images imgs

    imgs -- Torch dataset of (3xHxW) numpy images normalized in the range [-1, 1]
    cuda -- whether or not to run on GPU
    batch_size -- batch size for feeding into Inception v3
    splits -- number of splits
    """
    

    assert batch_size > 0
 
    # Set up dtype
    if cuda:
        dtype = torch.cuda.FloatTensor
    else:
        if torch.cuda.is_available():
            print("WARNING: You have a CUDA device, so you should probably set cuda=True")
        dtype = torch.FloatTensor

    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # Load inception model
    inception_model = inception_v3(pretrained=True, transform_input=False).type(dtype)
    inception_model.eval();
    up = nn.Upsample(size=(299, 299), mode='bilinear').type(dtype)
    def get_pred(x):
        if resize:
            x = up(x)
        x = inception_model(x)
        return F.softmax(x).data.cpu().numpy()

    # Get predictions
    preds = np.zeros((N, 1000))

    for i, data in enumerate(dataloader):
        batch_images = data['image'].type(dtype)
        batchv = Variable(batch_images)
        batch_size_i = batch_images.size()[0]

        preds[i*batch_size:i*batch_size + batch_size_i] = get_pred(batchv)

    # Now compute the mean kl-div
    split_scores = []

    for k in range(splits):
        part = preds[k * (N // splits): (k+1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)

if __name__ == '__main__':
    
    real_images_inception_score = False
    generated_images_inception_score = True

    T, h, w = 10, 64, 64

    transform = transforms.Compose([
            transforms.Resize((h, w)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    if real_images_inception_score == True:
        # Set up dataloader
        data_file = '/media/lily/HDD8T/scene_generation/datasets/VG1.4/data/dataset_vocab_109_objnum_1_10_imgnum_100382_val.json'
        image_dir = '/media/lily/HDD8T/scene_generation/datasets/VG1.4/all_images/VG_100K'

        dataset = VGDataset(data_file, image_dir, T, h, w, transform)
        num_imgs = len(dataset.images)
    
    elif generated_images_inception_score == True:
        image_dir = '/media/lily/HDD8T/scene_generation/sg2im/output_batch_img64'
        csv_file = './data/VG_generated_imgs.csv'

        dataset = GeneratedVGDataset(csv_file, image_dir, transform)
        num_imgs = len(dataset.dataset_frame)
    else:
        raise Exception("currently only calcuate the inception score for real images or generated images")

    print("number of images: ", num_imgs)
    print ("Calculating Inception Score...")
    print (inception_score(dataset =dataset, N = num_imgs, cuda=True, batch_size=64, resize=True, splits=10))

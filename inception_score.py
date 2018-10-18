import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
import torch.utils.data

from torchvision.models.inception import inception_v3

import numpy as np
from scipy.stats import entropy
from org_img_dataloader import *

def inception_score(cuda=True, batch_size=32, resize=False, splits=1):
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

    # Set up dataloader
    data_file = '/media/lily/HDD8T/scene_generation/datasets/VG1.4/data/dataset_vocab_109_objnum_1_10_imgnum_100382_val.json'
    image_dir = '/media/lily/HDD8T/scene_generation/datasets/VG1.4/all_images/VG_100K'

    T, h, w = 10, 64, 64
    transform = transforms.Compose([
        transforms.Resize((h, w)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    dataset = VGDataset(data_file, image_dir, T, h, w, transform)

    N = len(dataset.images)
    print("number of images: ", N)
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
    
    
    print ("Calculating Inception Score...")
    print (inception_score(cuda=True, batch_size=64, resize=True, splits=10))

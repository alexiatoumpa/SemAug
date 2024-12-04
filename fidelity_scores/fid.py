"""
Computation of the Frechet Inception Distance (FID) between two RGB images.
"""

from PIL import Image
import numpy as np
from scipy.linalg import sqrtm
import torch
from torchvision import models, transforms
import pdb

def preprocess_image(image, size=(299,299)):
    transform = transforms.Compose([transforms.Resize(size), transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                         std=[0.229, 0.224, 0.225])
                                    ])
    return tranform(image).unsqueeze(0)


def calculate_fid_score(image1path, image2path):
    # load images
    image1 = Image.open(image1path).convert("RGB")
    image2 = Image.open(image2path).convert("RGB")

    # using Inception model to get high-level feature representation
    model = models.inception_v3(pretrained=True, transform_input=False)
    model.fc = torch.nn.Identity()
    model.eval()

    image1 = preprocess_image(image1)
    image2 = preprocess_image(image2)

    with torch.no_grad():
        feature1 = model(image1).nummpy().squeeze()
        feature2 = model(image2).numpy().squeeze()

    # image1 = np.array(image1)
    # image2 = np.array(image2)
    # image1 = np.squeeze(image1)
    # image2 = np.squeeze(image2)
    # print(image1.shape, image2.shape) # (32, 32, 3)
    print(feature1.shape, feature2.shape)
    pdb.set_trace()

    # compute mean and covariance for the two images
    mean1 = np.mean(feature1, axis=0)
    mean2 = np.mean(feature2, axis=0)
    cov1 = np.cov(feature1, rowvar=False)
    cov2 = np.cov(feature2, rowvar=False)

    # compute FID score
    mean_diff = np.sum(mean1-mean2) ** 2
    cov_mean = sqrtm(cov1.dot(cov2))
    if np.iscomplexobj(cov_mean):
        cov_mean = cov_mean.real
    fid_score = mean_diff + np.trace(cov1 + cov2 - 2 * cov_mean)
    print(f"FID score: {fid_score}")

    return fid_score



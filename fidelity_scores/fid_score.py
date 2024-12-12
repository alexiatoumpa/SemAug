"""
Computation of the Frechet Inception Distance (FID) between two RGB images.
"""

import cv2
from PIL import Image
import numpy as np
from scipy.linalg import sqrtm
import torch
from torchvision import models, transforms

def preprocess_image(image, size=(299,299)):
    transform = transforms.Compose([transforms.Resize(size), transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                         std=[0.229, 0.224, 0.225])
                                    ])
    return transform(image).unsqueeze(0)


def calculate_fid_score(image1path, image2path):
    # load images
    image1 = cv2.imread(image1path, cv2.IMREAD_COLOR)
    image2 = cv2.imread(image2path, cv2.IMREAD_COLOR)

    # Convert OpenCV image to Pillow image
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
    image1 = Image.fromarray(image1)
    image2 = Image.fromarray(image2)

    # using Inception model to get high-level feature representation
    model = models.inception_v3(pretrained=True, transform_input=False)
    model.fc = torch.nn.Identity()
    model.eval()

    image1 = preprocess_image(image1)
    image2 = preprocess_image(image2)

    with torch.no_grad():
        feature1 = model(image1).numpy().squeeze()
        feature2 = model(image2).numpy().squeeze()

    # image1 = np.array(image1)
    # image2 = np.array(image2)
    # image1 = np.squeeze(image1)
    # image2 = np.squeeze(image2)
    # print(image1.shape, image2.shape) # (32, 32, 3)
    # print(feature1.shape, feature2.shape) # (2048,) (2048,)

    # compute mean and covariance for the two images
    mean1 = np.mean(feature1, axis=0)
    mean2 = np.mean(feature2, axis=0)
    cov1 = np.cov(feature1, rowvar=False)
    cov2 = np.cov(feature2, rowvar=False)

    # compute FID score
    mean_diff = np.sum(mean1-mean2) ** 2
    # cov_mean = sqrtm(cov1.dot(cov2))
    cov_mean = np.sqrt(cov1.dot(cov2))
    if np.iscomplexobj(cov_mean):
        cov_mean = cov_mean.real
    # fid_score = mean_diff + np.trace(cov1 + cov2 - 2 * cov_mean)
    fid_score = mean_diff + (cov1 + cov2 - 2 * cov_mean)
    print(f"FID score: {fid_score}")

    return fid_score


if __name__ == "__main__":

    initial_image_path = "./data/cifar/Augmented/101.jpg"
    aug_inpaint_categ_image_path = "./data/cifar/Augmented/Inpainting/airplane/0101.jpg"
    aug_erase_categ_image_path = "./data/cifar/Augmented/Erasing/airplane/0101.jpg"
    aug_noise_categ_image_path = "./data/cifar/Augmented/Noise/airplane/0101.jpg"

    inpaint_score = calculate_fid_score(initial_image_path, aug_inpaint_categ_image_path)
    erase_score = calculate_fid_score(initial_image_path, aug_erase_categ_image_path)
    noise_score = calculate_fid_score(initial_image_path, aug_noise_categ_image_path)
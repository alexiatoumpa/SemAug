"""
Computation of the Frechet Inception Distance (FID) between two RGB images.
"""

from PIL import Image
import numpy as np
from scipy.linalg import sqrtm

def calculate_fid_score(image1path, image2path):
    # load images
    image1 = Image.open(image1path).convert("RGB")
    image2 = Image.open(image2path).convert("RGB")

    image1 = image1.numpy().squeeze()
    image2 = image2.numpy().squeeze()

    # compute mean and covariance for the two images
    mean1 = np.mean(image1, axis=0)
    mean2 = np.mean(image2, axis=0)
    cov1 = np.cov(image1, rowvar=False)
    cov2 = np.cov(image2, rowvar=False)

    # compute FID score
    mean_diff = np.sum(mean1-mean2) ** 2
    cov_mean = sqrtm(cov1.dot(cov2))
    if np.iscomplexobj(cov_mean):
        cov_mean = cov_mean.real
    fid_score = mean_diff + np.trace(cov1 + cov2 - 2 * cov_mean)

    return fid_score



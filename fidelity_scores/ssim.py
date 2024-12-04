"""
Computation of the Structural Similarity Index (SSIM) between two RGB images.
"""

from PIL import Image
import numpy as np
from skimage.metrics import structural_similarity as ssim
import pdb

def calculate_ssim_score(image1path, image2path):
    # load images
    image1 = Image.open(image1path).convert("RGB")
    image2 = Image.open(image2path).convert("RGB")

    if image1.shape != image2.shape:
        image2 = image2.resize(image1.shape, Image.ANTIALIAS)

    image1 = np.array(image1)
    image2 = np.array(image2)

    # compute the SSIM score for each channel of the image and then take the mean
    ssim_value = 0
    for c in range(image1.shape[-1]):
        ssim_value += ssim(image1[:, :, c], image2[:, :, c], data_range=image2.max()-image2.min())
    
    # compute SSIM score
    ssim_score = ssim_value / 3
    print(f"SSIM score: {ssim_score}")

    return ssim_score



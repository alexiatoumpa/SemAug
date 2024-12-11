"""
Computation of the Structural Similarity Index (SSIM) between two RGB images.
SSIM values are between -1 and 1. A value closer to 1 indicates high similarity.
"""

from PIL import Image
import numpy as np
from skimage.metrics import structural_similarity as ssim
import pdb

def calculate_ssim_score(image1path, image2path):
    # load images
    image1 = Image.open(image1path).convert("RGB")
    image2 = Image.open(image2path).convert("RGB")
    pdb.set_trace()

    if image1.size != image2.size:
        image2 = image2.resize(image1.size, Image.LANCZOS)

    image1 = np.array(image1)
    image2 = np.array(image2)

    # compute the SSIM score for each channel of the image and then take the mean
    ssim_value = 0
    for c in range(image1.shape[-1]):
        ssim_value += ssim(image1[:, :, c], image2[:, :, c], data_range=image2.max()-image2.min(), full=True)
    
    # compute SSIM score
    ssim_score = ssim_value / 3
    print(f"SSIM score: {ssim_score}")

    return ssim_score



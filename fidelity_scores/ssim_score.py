"""
Computation of the Structural Similarity Index (SSIM) between two RGB images.
SSIM values are between -1 and 1. A value closer to 1 indicates high similarity.
"""

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

def calculate_ssim_score(image1path, image2path):
    # load images
    image1 = cv2.imread(image1path, cv2.IMREAD_COLOR)
    image2 = cv2.imread(image2path, cv2.IMREAD_COLOR)

    if image1.shape != image2.shape:
        image2 = cv2.resize(image2, image1.shape[:2], interpolation= cv2.INTER_LINEAR)

    # compute the SSIM score for each channel of the image and then take the mean
    ssim_value = 0
    for c in range(image1.shape[-1]):
        ssim_value += ssim(image1[:, :, c], image2[:, :, c], data_range=image2.max()-image2.min())
    
    # compute SSIM score
    ssim_score = ssim_value / 3
    print(f"SSIM score: {ssim_score}")

    return ssim_score


if __name__ == "__main__":

    initial_image_path = "./data/cifar/Augmented/101.jpg"
    aug_inpaint_categ_image_path = "./data/cifar/Augmented/Inpainting/airplane/0101.jpg"
    aug_erase_categ_image_path = "./data/cifar/Augmented/Erasing/airplane/0101.jpg"
    aug_noise_categ_image_path = "./data/cifar/Augmented/Noise/airplane/0101.jpg"

    inpaint_score = calculate_ssim_score(initial_image_path, aug_inpaint_categ_image_path)
    erase_score = calculate_ssim_score(initial_image_path, aug_erase_categ_image_path)
    noise_score = calculate_ssim_score(initial_image_path, aug_noise_categ_image_path)



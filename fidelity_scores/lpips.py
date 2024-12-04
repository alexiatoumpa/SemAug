"""
Computation of the Learned Perceptual Image Patch Similarity (LPIPS) between two 
RGB images.
"""

from PIL import Image
import lpips
import numpy as np

def normalize_image(image):
    image = image.astype(np.float32)
    # normalize pixel values to [0,1]
    image /= 255.0
    # scale values to [-1,1]
    image = image * 2 - 1
    return image


def calculate_lpips_score(image1path, image2path):
    # load images
    image1 = Image.open(image1path).convert("RGB")
    image2 = Image.open(image2path).convert("RGB")

    image1 = np.array(image1)
    image2 = np.array(image2)

    # image values should be normalized to [-1, 1]
    image1 = normalize_image(image1)
    image2 = normalize_image(image2)

    # initialize the lpips function
    lpips_func = lpips.LPIPS(net='alex') # net: 'alex', 'vgg', 'squeeze'

    # compute LPIPS score
    lpips_score = lpips_func(image1, image2)
    print(f"LPIPS score: {lpips_score}")

    return lpips_score_score





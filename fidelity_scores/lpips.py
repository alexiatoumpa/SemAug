"""
Computation of the Learned Perceptual Image Patch Similarity (LPIPS) between two 
RGB images.
"""

from PIL import Image
import lpips
import torch

def calculate_ssim_score(image1path, image2path):
    # load images
    image1 = Image.open(image1path).convert("RGB")
    image2 = Image.open(image2path).convert("RGB")

    # initialize the lpips function
    lpips_func = lpips.LPIPS(net='alex') # net: 'alex', 'vgg', 'squeeze'

    # image values should be normalized to [-1, 1]
    
    # compute LPIPS score
    lpips_score = lpips_func(image1, image2)
    print(f"LPIPS score: {lpips_score}")

    return lpips_score_score



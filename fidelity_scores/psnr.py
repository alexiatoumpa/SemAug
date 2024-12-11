"""
Computation of the Peak Signal-to-Noise Ratio (PSNR) between two 
RGB images.
"""

from PIL import Image
import numpy as np

def normalize_image(image):
    image = image.astype(np.float32)
    # normalize pixel values to [0,1]
    image /= 255.0
    # scale values to [-1,1]
    image = image * 2 - 1
    return image


def calculate_psnr_score(image1path, image2path):
    # load images
    image1 = Image.open(image1path).convert("RGB")
    image2 = Image.open(image2path).convert("RGB")

    image1 = np.array(image1)
    image2 = np.array(image2)

    if image1.shape != image2.shape:
        image2 = image2.resize(image1.shape, Image.ANTIALIAS)

    # compute MSE error
    mse = np.mean((image1-image2) ** 2)

    # PSNR score is infinite for identical images
    if mse == 0:
        return float('inf')

    # compute PSNR score
    max_pixel = np.max(image2)
    psnr_score = 20 * np.log10(max_pixel / np.sqrt(mse))
    print(f"PSNR score: {psnr_score}")

    return psnr_score





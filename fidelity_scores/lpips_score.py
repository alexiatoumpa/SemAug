"""
Computation of the Learned Perceptual Image Patch Similarity (LPIPS) between two 
RGB images.
"""

import lpips
import numpy as np
import cv2

def normalize_image(image):
    image = image.astype(np.float32)
    # normalize pixel values to [0,1]
    image /= 255.0
    # scale values to [-1,1]
    image = image * 2 - 1
    return image


def calculate_lpips_score(image1path, image2path, net='alex'):
    # load images
    image1 = cv2.imread(image1path, cv2.IMREAD_COLOR)
    image2 = cv2.imread(image2path, cv2.IMREAD_COLOR)

    if image1.shape != image2.shape:
        image2 = cv2.resize(image2, image1.shape[:2], interpolation= cv2.INTER_LINEAR)

    # image values should be normalized to [-1, 1]
    image1 = normalize_image(image1)
    image2 = normalize_image(image2)

    # convert images to tensors
    image1 = lpips.im2tensor(image1)
    image2 = lpips.im2tensor(image2)

    # initialize the lpips function
    lpips_func = lpips.LPIPS(net=net) # net: 'alex', 'vgg', 'squeeze'

    # compute LPIPS score
    lpips_score = lpips_func(image1, image2).item()
    print(f"LPIPS score: {lpips_score}")

    return lpips_score


if __name__ == "__main__":

    initial_image_path = "./data/cifar/Augmented/101.jpg"
    aug_inpaint_categ_image_path = "./data/cifar/Augmented/Inpainting/airplane/0101.jpg"
    aug_erase_categ_image_path = "./data/cifar/Augmented/Erasing/airplane/0101.jpg"
    aug_noise_categ_image_path = "./data/cifar/Augmented/Noise/airplane/0101.jpg"

    inpaint_score = calculate_lpips_score(initial_image_path, aug_inpaint_categ_image_path, net='alex')
    erase_score = calculate_lpips_score(initial_image_path, aug_erase_categ_image_path, net='alex')
    noise_score = calculate_lpips_score(initial_image_path, aug_noise_categ_image_path, net='alex')





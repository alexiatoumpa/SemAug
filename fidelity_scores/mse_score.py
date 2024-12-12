"""
Computation of the Mean Squared Error (mse) between two RGB images.
"""

import cv2
import numpy as np

def calculate_mse_score(image1path, image2path):
    # load images
    image1 = cv2.imread(image1path, cv2.IMREAD_COLOR)
    image2 = cv2.imread(image2path, cv2.IMREAD_COLOR)

    if image1.shape != image2.shape:
        image2 = cv2.resize(image2, image1.shape[:2], interpolation= cv2.INTER_LINEAR)
    
    # compute MSE score
    mse_score = np.mean((image1 - image2) ** 2)
    print(f"MSE score: {mse_score}")

    return mse_score


if __name__ == "__main__":

    initial_image_path = "./data/cifar/Augmented/101.jpg"
    aug_inpaint_categ_image_path = "./data/cifar/Augmented/Inpainting/airplane/0101.jpg"
    aug_erase_categ_image_path = "./data/cifar/Augmented/Erasing/airplane/0101.jpg"
    aug_noise_categ_image_path = "./data/cifar/Augmented/Noise/airplane/0101.jpg"

    inpaint_score = calculate_mse_score(initial_image_path, aug_inpaint_categ_image_path)
    erase_score = calculate_mse_score(initial_image_path, aug_erase_categ_image_path)
    noise_score = calculate_mse_score(initial_image_path, aug_noise_categ_image_path)


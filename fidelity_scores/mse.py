"""
Computation of the Mean Squared Error (mse) between two RGB images.
"""

from PIL import Image
import numpy as np

def calculate_mse_score(image1path, image2path):
    # load images
    image1 = Image.open(image1path).convert("RGB")
    image2 = Image.open(image2path).convert("RGB")
    
    # compute MSE score
    mse_score = np.mean((image1 - image2) ** 2)
    print(f"MSE score: {mse_score}")

    return mse_score



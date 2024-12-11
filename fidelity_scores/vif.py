"""
Computation of the Visual Information Fidelity (VIF) score between two RGB images.
"""

from PIL import Image
import numpy as np
from sewar.full_Ref import vifp

def calculate_vif_score(image1path, image2path):
    # load images
    image1 = Image.open(image1path).convert("L")
    image2 = Image.open(image2path).convert("L")

    image1 = np.array(image1)
    image2 = np.array(image2)

    # compute VIF score
    vif_score = vifp(image1, image2)
    print(f"VIFscore: {vif_score}")

    return psnr_score





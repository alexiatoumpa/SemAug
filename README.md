# SemAug

## Requirements
In a python virtual environment install the following pip packages:
```bash
pip install torch
pip install --upgrade diffusers[torch]
pip install tensorflow
pip install matplotlib
pip install pandas
pip install timm
pip install opencv-python
pip install scikit-image
pip install nltk
pip install spacy
pip install transformers

pip install lpips
pip install sewar
```
Then follow the steps below:
1. Download the Mask R-CNN models from [here](https://github.com/sambhav37/Mask-R-CNN/tree/master/mask-rcnn-coco).
2. Create an account in hugging face and add token in `InpaintingDiffusionModel.py` stored in the variable `access_token`.
3. Download the `inpainting.py` file from hugging face and store it under the `augmentation/` directory:
```bash
pip freeze | grep diffusers
wget https://raw.githubusercontent.com/huggingface/diffusers/main/examples/inference/inpainting.py
```

## Run code
```bash
python run_semaug.py \
--dataset <path_to_test_dataset>
```
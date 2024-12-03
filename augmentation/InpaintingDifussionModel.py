from diffusers import StableDiffusionInpaintPipeline

from PIL import Image
import torch

HF_DATASETS_OFFLINE = 1
TRANSFORMERS_OFFLINE = 1


def Inpainting(Init_img, mask, augmented_caption):

        access_token = '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' 
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #
        # if device.type != 'cuda':
        #     raise ValueError("need to run on GPU")
        device = "cpu" #"cuda" ##
        # Old model for inpainring
        # pipe = StableDiffusionInpaintPipeline.from_pretrained(
        #     "CompVis/stable-diffusion-v1-4", revision="fp16",
        #     # torch_dtype=torch.float16,
        #     use_auth_token=access_token).to(device)
        
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
              "runwayml/stable-diffusion-inpainting", #torch_dtype=torch.float16
              )
        pipe = pipe.to(device)

        init_image = Image.open(Init_img).resize((512, 512))
        mask_image = Image.open(mask).resize((512, 512))

        with torch.no_grad(): 
            images = pipe(prompt=augmented_caption, image=init_image, mask_image=mask_image, strength=0.75).images

        return images


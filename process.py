from augmentation.Masking import *
from augmentation.InpaintingDifussionModel import Inpainting
from nlp.Caption_Enrichement_NLP import caption_category, change_caption
from fidelity_scores.fid_score import calculate_fid_score
from fidelity_scores.ssim_score import calculate_ssim_score
from fidelity_scores.mse_score import calculate_mse_score
from fidelity_scores.lpips_score import calculate_lpips_score
from fidelity_scores.psnr_score import calculate_psnr_score
from fidelity_scores.vif_score import calculate_vif_score

# import re
from timm.data.random_erasing import RandomErasing
from torchvision import transforms
from matplotlib import pyplot as plt
import random as ran
import skimage.io as io
import urllib.request
import urllib
import cv2
import os
import spacy
from tqdm import tqdm
import numpy as np
from pathlib import Path
import argparse
from skimage import color
from skimage import io
# import skimage.color
import skimage.filters
from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


def get_caption(label,category):
    Initial_caption = category[label]
    Initial_caption = Initial_caption
    return Initial_caption


# def augment_from_folder(seed_size,approach, directory_aug_data, folder_dir, categories,ImgsIds, categories_keys):
#     scores = []
#     images = []
#     inpainting_testing_data=[]
#     erasing_testing_data = []
#     noise_testing_data = []
#     testing_data=[]
#     Captions=get_img_caption(ImgsIds)
#     augmented_path_inpaint="./data/coco/Augmented/Inpainting/"
#     augmented_path_erase = "./data/coco/Augmented/Erasing/"
#     augmented_path_noise = "./data/coco/Augmented/Noise/"
#     if approach == 'Inpainting':
#         for c in categories:
#             print('label',c)
#             label=c
#             path = os.path.join(folder_dir, c)
#             class_num = categories.index(c)
#             count = 0

#             for i in os.listdir(path):
#                 count = count + 1
#                 print("file path",i)
#                 ini_path=os.path.join(path, i)
#                 try:
#                     img_array = cv2.imread(os.path.join(path, i))
#                     # img_array=cv2.resize(img_array,(128,128))
#                     testing_data.append([img_array, class_num])
#                 except Exception as e:
#                     pass

#                 height, width, _ = img_array.shape
#                 img_ini = img_array
#                 mask_path = os.path.join(directory_aug_data, "Masking/" + i)
#                 Masking = Create_Mask(img_ini, mask_path)
#                 W_mask = Masking.get_mask()
#                 mask = cv2.cvtColor(W_mask, cv2.COLOR_BGR2RGB)
#                 plt.imsave(mask_path, mask)
#                 print("saved mask")
#                 id =int(deleteLeadingZeros(i))
#                 # print("id",id)
#                 if id in Captions.keys():
#                     Initial_caption = Captions[id]["caption"]
#                     print("Initial caption", Initial_caption)
#                 rep = 0
#                 for cat in categories:
#                     if cat != label:
#                         print(cat)
#                         current_path=os.path.join(augmented_path_inpaint,cat)
#                         if not os.path.exists(current_path):
#                             os.mkdir(current_path)
#                         if str(label) in str(Initial_caption).lower():
#                             aug_caption= change_caption(nlp,Initial_caption,label,"animal",cat)
#                             print("aug:: ",aug_caption)
#                         else:
#                             aug_caption=cat
#                             print("cap:: ", aug_caption)
#                         class_nbr = categories.index(cat)
#                         Aug_Imgs = Inpainting(ini_path, mask_path, aug_caption)
#                         Aug_Img = Aug_Imgs[0]
#                         plt.imshow(Aug_Img)
#                         plt.axis('off')
#                         inpaint = (str(rep) + str(i))
#                         inpaint_path = os.path.join(current_path, inpaint)
#                         # print("TYPE Aug", type(Aug_Img))£#PIL image
#                         Aug_Img_save = numpy.array(Aug_Img)
#                         Aug_Img_save = cv2.resize(Aug_Img_save, (height, width))
#                         cv2.imwrite(inpaint_path, Aug_Img_save)
#                         plt.imsave(inpaint_path, Aug_Img_save)
#                         inpainting_testing_data.append([Aug_Img_save, class_nbr])
#                         print("aug saved")
#                         # rep=rep+1
#                         clip_score = p_get_clip_score(Aug_Img, Initial_caption, aug_caption)
#                         print("the threshold score", clip_score.item())

#                         # erase#######
#                         img = Image.open(ini_path)
#                         x = transforms.ToTensor()(img)
#                         random_erase = RandomErasing(probability=1, mode='pixel', device='cpu')
#                         aug = random_erase(x).permute(1, -1, 0)
#                         plt.imshow(np.squeeze(aug))
#                         # clip_erase = p_get_clip_score(np.squeeze(aug), Initial_caption, label)
#                         # print("the erase clip score", clip_erase.item())
#                         plt.axis('off')
#                         current_erasepath = os.path.join(augmented_path_erase, cat)
#                         if not os.path.exists(current_erasepath):
#                             os.mkdir(current_erasepath)
#                         erase_pa= (str(rep) + str(i))
#                         erase_path = os.path.join(current_erasepath, erase_pa)
#                         plt.savefig(erase_path)
#                         erasing_testing_data.append([aug, class_num])
#                         current_noisepath = os.path.join(augmented_path_noise, cat)
#                         if not os.path.exists(current_noisepath):
#                             os.mkdir(current_noisepath)
#                         noise_pa = (str(rep) + str(i))
#                         noise_path = os.path.join(current_noisepath, noise_pa)
#                         im = cv2.imread(ini_path, 0)
#                         mean = 0
#                         std = 1
#                         gaus_noise = np.random.normal(mean, std, im.shape)
#                         image = im.astype("int16")
#                         noise_img = image + gaus_noise
#                         # noise_img = np.array(noise_img)
#                         # clip_noise = p_get_clip_score(noise_img, Initial_caption, label)
#                         # print("the noise clip score", clip_noise.item())
#                         noise_img = cv2.resize(noise_img, (height, width))
#                         plt.imshow(noise_img)
#                         plt.imsave(noise_path, noise_img)
#                         noise_testing_data.append([noise_img, class_num])
#                         SSIM_N, FID_Noise = get_scores(ini_path, noise_path, height, width)
#                         SSIM_inpaint, FID_inpainting = get_scores(ini_path, inpaint_path, height, width)
#                         SSIM_E, FID_Erase = get_scores(ini_path, erase_path, height, width)
#                         line = [str(i), SSIM_inpaint, SSIM_E, FID_inpainting, FID_Erase, SSIM_N, FID_Noise, clip_score.item()]
#                         scores.append(line)
#                         images.append([str(id), Initial_caption, aug_caption,ini_path,inpaint_path, erase_path,noise_path,label,cat])

#     return scores, images,inpainting_testing_data,erasing_testing_data,noise_testing_data


# '''
# augment_coco_images works perfectly and rely on mask rnn to create masks !
# '''
# def augment_coco_images(seed_size=42, data_directory_path='./', dataset_path='./'):
#     results = []
#     images = []
#     nlp = spacy.load("en_core_web_sm")
#     datapath = dataset_path + "/annotations/instances_train2017.json"
#     KeyObjFile = dataset_path + "/annotations/person_keypoints_train2017.json"
#     CaptionFile = dataset_path + "/annotations/captions_train2017.json"
#     category = 'dog'
#     subcategory = 'person'
#     rep = 'panda'
#     filterClasses3 = [category, subcategory]
#     pre = Datapipline(datapath, KeyObjFile, CaptionFile) ##??
#     maincategories, subcategories = pre.load_cat_info()
#     Initial_seed_Set, Initial_seed_Ids = pre.select_subCatg(filterClasses3)
#     next_pix = Initial_seed_Set
#     ran.shuffle(next_pix)
#     coco_kps = pre.KObj
#     coco_caps = pre.Caps

#     # Load_images_with keypoints objects and captions
#     for i, img_path in enumerate(next_pix[0:seed_size]):
#         img = pre.coco.loadImgs(img_path)[0]
#         id = str(img['id'])
 
#         if len(img.shape) != 3:
#             print("shape")
#             print(img.shape)
#             continue
#         else:
#             I = io.imread(img['coco_url'])

#             ini_path = os.path.join(data_directory_path, "Initial2/I_" + id + ".jpeg")
#             mask_path = os.path.join(data_directory_path, "masks/M_" + id + ".jpeg")
#             erase_path = os.path.join(data_directory_path, "erase/E_" + id + ".jpeg")
#             noise_path = os.path.join(data_directory_path, "noise/N_" + id + ".jpeg")
#             img_pil = PIL.Image.open(urllib.request.urlopen(img['coco_url']))
#             annIds = coco_kps.getAnnIds(imgIds=img['id'], catIds=Initial_seed_Ids, iscrowd=None)
#             anns = coco_kps.loadAnns(annIds)

#             CaptionIds = coco_caps.getAnnIds(imgIds=img['id'])
#             caption = coco_caps.loadAnns(CaptionIds)
#             if category not in str(caption[0]['caption']).lower():
#                 continue
#             else:
#                 Initial_caption = caption[0]['caption']
#                 plt.imsave(ini_path, I)
#                 aug_caption_Category = caption_category(nlp, Initial_caption, category, subcategory, rep)
#                 """
#                     Augmentation process
#                 """
#                     # 1/ create masque
#                 img = Image.open(ini_path)

#                 Masking = Create_Mask(I, mask_path)
#                 W_mask = Masking.get_mask()
#                 if type(W_mask)==np.ndarray :

#                     mask = cv2.cvtColor(W_mask, cv2.COLOR_BGR2RGB)
#                     plt.imsave(mask_path, mask)
#                     print("saved mask")
#                     Aug_Imgs = Inpainting(ini_path, mask_path, aug_caption_Category)
#                     Aug_Img = Aug_Imgs[0]
#                     plt.imshow(Aug_Img)
#                     plt.axis('off')
#                     inpaint_path = os.path.join(data_directory_path,"Inpaint/In_" + id + ".jpeg")
#                     plt.savefig(inpaint_path)
#                     clip_score = p_get_clip_score(Aug_Img, Initial_caption, aug_caption_Category)
#                     if clip_score < 70:
#                         print("the score:", clip_score.item())
#                         continue
#                     else:
#                         print("the score", clip_score.item())
#                         plt.savefig(inpaint_path)
#                         img = Image.open(ini_path)
#                         x = transforms.ToTensor()(img)
#                         random_erase = RandomErasing(probability=1, mode='pixel', device='cpu')
#                         aug = random_erase(x).permute(1, -1, 0)
#                         plt.imshow(np.squeeze(aug))
#                         plt.axis('off')
#                         plt.savefig(erase_path)
#                         # Gaussian Noise

#                         im = cv2.imread(ini_path, 0)
#                         mean = 0
#                         std = 1
#                         gaus_noise = np.random.normal(mean, std, im.shape)
#                         image = im.astype("int16")
#                         noise_img = image + gaus_noise
#                         plt.imshow(noise_img)
#                         plt.imsave(noise_path, noise_img)
#                         SSIM_inpaint, SSIM_E, SSIM_N, FID_inpainting, FID_Erase, FID_Noise = get_scores(ini_path,
#                                                                                                         inpaint_path,
#                                                                                                         erase_path,
#                                                                                                         noise_path)
#                         line = [str(id), SSIM_inpaint, SSIM_E, SSIM_N, FID_inpainting, FID_Erase, FID_Noise,
#                                 clip_score.item()]
#                         results.append(line)
#                         images.append({"id": str(id), "caption": Initial_caption, "Aug_caption": aug_caption_Category,
#                                        "original": ini_path, "Inpainting": inpaint_path, "Erase": erase_path,
#                                        "Noise": noise_path})
#                 else:
#                     continue
#     return results,images


# def getClassName(classID, cats):
#     for i in range(len(cats)):
#         if cats[i]['id'] == classID:
#             return cats[i]['name']
#     return "None"


def create_aug_data_directories(data_directory_path='./', subdir=["Mask", "Inpainting", 
    "Erasing", "Noise"]):

    aug_subdir_paths = []
    if isinstance(subdir, list):
        for d in subdir:
            path_ = os.path.join(data_directory_path, d)
            if not os.path.exists(path_):
                os.mkdir(path_)
            aug_subdir_paths.append(path_)
    elif isinstance(data_directory_path, list):
        for d in data_directory_path:
            path_ = os.path.join(d, subdir)
            if not os.path.exists(path_):
                os.mkdir(path_)
            aug_subdir_paths.append(path_)   
    return aug_subdir_paths


def create_mask_image(image=None, mask_img_path='./'):
    Masking = Create_Mask(image, mask_img_path)
    W_mask = Masking.get_mask()
    mask = cv2.cvtColor(W_mask, cv2.COLOR_BGR2RGB)
    return mask


def create_inpaint_image(image_path='./', mask_path='./', caption='cat'):
    inpainted_images = Inpainting(image_path, mask_path, caption)
    inpaint_image = inpainted_images[0]    
    inpaint_image = np.array(inpaint_image)
    return inpaint_image


def augment_cifar_images(x_test, y_test, seed_size=42, data_directory_path='./', categories=[]):
    scores = []
    images = []

    [aug_mask_path, aug_inpaint_path, aug_erase_path, aug_noise_path] = \
        create_aug_data_directories(data_directory_path=data_directory_path,
                                    subdir=["Mask", "Inpainting", "Erasing", "Noise"])
    
    id = 100
    for features, label in zip(x_test[:seed_size], y_test[:seed_size]):
        id += 1
        print(id)

        # read initial image
        initial_image_path = os.path.join(data_directory_path, str(id) + ".jpg")

        

        height, width, _ = features.shape
        initial_image = features
        # save original image
        cv2.imwrite(initial_image_path, initial_image)
        print("label", np.argmax(label))

        image = Image.open(initial_image_path) # used in erase
        imageT = transforms.ToTensor()(image) # used in erase

        image = cv2.imread(initial_image_path, 0) # used in noise

        # create mask
        aug_mask_image_path = os.path.join(aug_mask_path, str(id) + ".jpg")
        mask = create_mask_image(image=initial_image, mask_img_path=aug_mask_image_path)
        # save mask image
        cv2.imwrite(aug_mask_image_path, mask)

        initial_caption = get_caption(label[0], categories)
        print("Initial caption: ", initial_caption)
        rep = 0 ## TO CHECK: this doesn't increase
        subcategory = ''

        # parse through all the categories to create an augmentation image
        for category in categories:
            if category != initial_caption:
                print("Category: ", category)
                # create the directories for each category
                [aug_inpaint_categ_path, aug_erase_categ_path, aug_noise_categ_path] = \
                    create_aug_data_directories(data_directory_path=[aug_inpaint_path, 
                                                                     aug_erase_path, 
                                                                     aug_noise_path], 
                                                subdir=category)

                # create new caption for the specific category
                aug_caption_category = caption_category(initial_caption, category, subcategory)
                
                # Inpainting
                aug_inpaint_categ_image_path = os.path.join(aug_inpaint_categ_path, str(rep) + str(id) + ".jpg")
                inpaint_image = create_inpaint_image(image_path=initial_image_path, mask_path=aug_mask_image_path, 
                                                     caption=aug_caption_category)
                # save inpaint image
                # inpaint_image = cv2.resize(inpaint_image, (height, width))
                cv2.imwrite(aug_inpaint_categ_image_path, inpaint_image)

                # Erase
                random_erase = RandomErasing(probability=1, mode='pixel', device='cpu')
                erased_image = random_erase(imageT).permute(1, -1, 0)
            
                aug_erase_categ_image_path = os.path.join(aug_erase_categ_path, str(rep) + str(id) + ".jpg")
                # save erased image
                plt.imshow(np.squeeze(erased_image))
                plt.axis('off')
                plt.savefig(aug_erase_categ_image_path)

                # Noise
                mean, std = 0, 1
                gaussian_noise = np.random.normal(mean, std, image.shape)
                image = image.astype("int16")
                noise_image = image + gaussian_noise
                # noise_image = cv2.resize(noise_image, (height, width))
                
                aug_noise_categ_image_path = os.path.join(aug_noise_categ_path, str(rep) + str(id) + ".jpg")
                # save noise image
                cv2.imwrite(aug_noise_categ_image_path, noise_image)

                # Calculate fidelity scores:
                # FID
                FID_inpaint = calculate_fid_score(initial_image_path, aug_inpaint_categ_image_path)
                FID_erase = calculate_fid_score(initial_image_path, aug_erase_categ_image_path)
                FID_noise = calculate_fid_score(initial_image_path, aug_noise_categ_image_path)
                # SSIM
                SSIM_inpaint = calculate_ssim_score(initial_image_path, aug_inpaint_categ_image_path)
                SSIM_erase = calculate_ssim_score(initial_image_path, aug_erase_categ_image_path)
                SSIM_noise = calculate_ssim_score(initial_image_path, aug_noise_categ_image_path)
                # LPIPS AlexNet
                LPIPS_inpaint_alex = calculate_lpips_score(initial_image_path, aug_inpaint_categ_image_path, net='alex')
                LPIPS_erase_alex = calculate_lpips_score(initial_image_path, aug_erase_categ_image_path, net='alex')
                LPIPS_noise_alex = calculate_lpips_score(initial_image_path, aug_noise_categ_image_path, net='alex')
                # LPIPS VGG
                LPIPS_inpaint_vgg = calculate_lpips_score(initial_image_path, aug_inpaint_categ_image_path, net='vgg')
                LPIPS_erase_vgg = calculate_lpips_score(initial_image_path, aug_erase_categ_image_path, net='vgg')
                LPIPS_noise_vgg = calculate_lpips_score(initial_image_path, aug_noise_categ_image_path, net='vgg')
                # LPIPS squeeze
                LPIPS_inpaint_sq = calculate_lpips_score(initial_image_path, aug_inpaint_categ_image_path, net='squeeze')
                LPIPS_erase_sq = calculate_lpips_score(initial_image_path, aug_erase_categ_image_path, net='squeeze')
                LPIPS_noise_sq = calculate_lpips_score(initial_image_path, aug_noise_categ_image_path, net='squeeze')
                # MSE
                MSE_inpaint = calculate_mse_score(initial_image_path, aug_inpaint_categ_image_path)
                MSE_erase = calculate_mse_score(initial_image_path, aug_erase_categ_image_path)
                MSE_noise = calculate_mse_score(initial_image_path, aug_noise_categ_image_path)
                # PSNR
                PSNR_inpaint = calculate_psnr_score(initial_image_path, aug_inpaint_categ_image_path)
                PSNR_erase = calculate_psnr_score(initial_image_path, aug_erase_categ_image_path)
                PSNR_noise = calculate_psnr_score(initial_image_path, aug_noise_categ_image_path)
                # VIF
                VIF_inpaint = calculate_vif_score(initial_image_path, aug_inpaint_categ_image_path)
                VIF_erase = calculate_vif_score(initial_image_path, aug_erase_categ_image_path)
                VIF_noise = calculate_vif_score(initial_image_path, aug_noise_categ_image_path)

                scores.append([str(id), initial_caption, aug_caption_category, 
                               "inpaint", MSE_inpaint, PSNR_inpaint, FID_inpaint, 
                               SSIM_inpaint, LPIPS_inpaint_alex, LPIPS_inpaint_vgg, 
                               LPIPS_inpaint_sq, VIF_inpaint,
                               ])
                scores.append([str(id), initial_caption, aug_caption_category, 
                               "erase", MSE_erase, PSNR_erase, FID_erase, SSIM_erase, 
                               LPIPS_erase_alex, LPIPS_erase_vgg, LPIPS_erase_sq, VIF_erase,
                               ])
                scores.append([str(id), initial_caption, aug_caption_category, 
                               "noise", MSE_noise, PSNR_noise, FID_noise, SSIM_noise, 
                               LPIPS_noise_alex, LPIPS_noise_vgg, LPIPS_noise_sq, VIF_noise,
                               ])

                # images.append([str(id), initial_caption, aug_caption_category, 
                #     initial_image_path, aug_inpaint_categ_image_path, 
                #     aug_erase_categ_image_path, aug_noise_categ_image_path, 
                #     label, category])
                images.append([str(id), initial_caption, aug_caption_category, 
                    "inpaint", initial_image_path, aug_inpaint_categ_image_path, 
                    label, category])
                images.append([str(id), initial_caption, aug_caption_category, 
                    "erase", initial_image_path, aug_erase_categ_image_path, 
                    label, category])
                images.append([str(id), initial_caption, aug_caption_category, 
                    "noise", initial_image_path, aug_noise_categ_image_path, 
                    label, category])

    return images, scores



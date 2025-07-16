from augmentation.InpaintingDifussionModel import Inpainting
from process import augment_images, augment_cifar_images
from nlp.Caption_Enrichement_NLP import *

import numpy as np
import torch
from matplotlib import pyplot as plt
# from tensorflow.keras.datasets import cifar10
from utils.datasets import (load_cifar10_dataset, load_custom_dataset, 
                            load_test_imagepaths, load_imagenet_dataset
)
import csv
import os
import argparse
from datetime import datetime
import pdb



__version__ = 0.2
def parse_arguments():
    """
    Parse command line argument and construct the DNN
    :return: a dictionary comprising the command-line arguments
    """
    text = 'Data Augmentation pipline'

    # initiate the parser
    parser = argparse.ArgumentParser(description=text)

    # new command-line arguments
    parser.add_argument("-V", "--version",  help="show program version",
                        action="version", version="DeepFault %f" % __version__)
    parser.add_argument("-M", "--methodology", help="semantic augmentation methodology", 
                        choices=['Inpainting','Imagic'])
    parser.add_argument("-DS", "--dataset", default="cifar10", help="The dataset to be used (imagenet, cifar10, or leaf).", 
                        choices=["imagenet","cifar10","leaf"])
    parser.add_argument("-PG", "--path2graph", help="Path to the mask prediction model graph file.", 
                        type=str)
    parser.add_argument("-PW", "--path2weights", help="Path to the mask prediction model weights file.", 
                        type=str)
    # parser.add_argument("-Me", "--measure", help="the approach to be employed \
    #                         to measure similarity", choices=['SSIM', 'FID'])

    parser.add_argument("-Cap", "--Augmented_caption", help="the image caption")
    parser.add_argument("-K", "--iteration", help="nbre of iteration for augmenting an image.", 
                        type=int)
    parser.add_argument("-SS", "--seed_size", default=2, help="size of initial set of seed images.", 
                        type=int)
    parser.add_argument("-LOG", "--logfile", default='DataAugment.log', help="path to log file")

    parser.add_argument("-FS", "--fidelity", action='store_true', help="Bollean argument for computing the fidelity scores.")

    args = parser.parse_args()

    return vars(args)


# def save_image(Aug_img, base_file, approach):
#     directory='./data/Augmented'
#     name = base_file
#     new_file = '{}''_''{}'+'.png'.format(name, approach)
#     completeName =os.path.join(directory,new_file )
#     print (type(Aug_img))
#     plt.imsave(name, Aug_img)
#     print("saved")


if __name__ == "__main__":

    args = parse_arguments()
    print(args)
    approach = args['methodology'] if args['methodology'] else 'Inpainting'

    # nlp = spacy.load("en_core_web_sm")
    # taxonomy = ['smiling', 'waving', 'talking', 'sleeping', 'siting', 'laughting', 
    #             'jumping', 'wearing a mask']
    # measure = args['measure'] if not args['measure'] == None else 'SSIM'
    # caption = args['intial_caption'] if not args['intial_caption'] == None else 'a person'
    Augmented_caption = args['Augmented_caption'] if not args['Augmented_caption'] == None \
        else 'a person'
    seed_size = args['seed_size']
    dataset = args['dataset']
    path2graph=args['path2graph'] if 'path2graph' in args else "/home/alexiatoumpa/dev/Github/SemAug/models/frozen_inference_graph.pb"
    path2weights=args['path2weights'] if 'path2weights' in args else "/home/alexiatoumpa/dev/Github/SemAug/models/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt"
    # datatype = args['datatype'] if not args['datatype'] == None else 'cifar'
    logfile_name = args['logfile']
    logfile = open(logfile_name, 'a')
    extension='.csv'

    iteration="_11_"
    # Format as DATE - REGION - REPORT TYPE
    start_time = datetime.now()
    results=[]
    fidelity_analysis = args['fidelity']
    # line = [str(id), SSIM_inpaint, SSIM_E, SSIM_N, FID_inpainting, FID_Erase, FID_Noise, clip_score.item()]
    # results.append(line)
    # images.append({"id": str(id), "caption": Initial_caption, "Aug_caption": aug_caption_Category, "original": ini_path,
    #                "Inpainting": inpaint_path, "Erase": erase_path, "Noise": noise_path})

    entete_results=['img_id', 'caption', 'augmented_caption','method', 'MSE', 
                    'PSNR','FID', 'SSIM', 'LPIPS_alex', 'LPIPS_vgg', 'LPIPS_squeeze', 'VIF']

    # device = torch.device("cpu")
    # entete_imgs=['id', 'caption', 'augmented_caption', 'original_img', 'inpainting', 
    #              'erase', 'noise', 'label', 'augmentation_category']
    entete_imgs=['id', 'caption', 'augmented_caption', 'method', 'original_img', 'augmented_img', 
                 'label', 'augmentation_category']
    approach = 'Inpainting'

    date = datetime.strftime(datetime.now(), '%Y-%m-%d')
    info = {f'{date} - {approach} - {dataset} - {extension}'}
    extension = '.csv'
    k="scores_matadata"
    a="augmentedset"
    file_name = f'{date}-{approach}-{dataset}-{k}-{iteration}-{extension}'
    file_images=f'{date}-{approach}-{dataset}-{a}-{iteration}-{extension}'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # with torch.cuda.amp.autocast(True):
    #####dataset and input
    start_time = datetime.now()
    logfile_name = './results/DataAugment.log'
    logfile = open(logfile_name, 'a')

    print(dataset,"loading... ... ...")
    if dataset=='cifar10':
        categories = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 
                    'horse', 'ship', 'truck']
        data_directory_path = "./data/cifar/Augmented/"

        # Load dataset
        # (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        (x_train, y_train), (x_test, y_test) = load_cifar10_dataset()
        print("x_test shape:", x_test.shape)
        print("y_test shape:", y_test.shape)
        print(type(x_test))
        print(type(y_test))
        # Create augmented data
        # images, scores = augment_cifar_images(x_test, y_test, dataset=dataset,
        #                                 seed_size=seed_size, data_directory_path=data_directory_path, 
        #                                 categories=categories, path2graph=path2graph, path2weights=path2weights)
    elif dataset=='imagenet':
        categories = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 
                    'horse', 'ship', 'truck']
        data_directory_path = "./data/imagenet/Augmented/"
        # Load dataset
        dataset_path = '/home/alexiatoumpa/data/ImageNet/ILSVRC/Data/DET/test/'
        x_test = load_imagenet_dataset(dataset_path=dataset_path)
        y_test = [None for i in range(len(x_test))]  # Placeholder, as labels are not available in this dataset

        print("x_test shape:", len(x_test))
        print("x_test type:", type(x_test))

    elif dataset=='leaf':
        categories = ['yellowed leaf', 'rotten leaf', 'fungus', 'dehydrated leaf']
        data_directory_path = "./data/leaf/Augmented/"

        # Load dataset
        # dataset_path = '/home/alexiatoumpa/data/QDC/Grape Varieties_for image processing/'
        dataset_path = '/home/alexiatoumpa/data/grape_dataset/'
        # (x_train, y_train), (x_test, y_test) = load_custom_dataset(dataset_path=dataset_path)
        (_, _), (x_test, y_test) = load_test_imagepaths(dataset_path=dataset_path)

        try:
            print("x_test shape:", x_test.shape)
            print("y_test shape:", y_test.shape)
        except Exception:
            print("x_test shape:", len(x_test))
            print("y_test shape:", len(y_test))
        print("x_test type:", type(x_test))
        print("y_test type:", type(y_test))

    # Create augmented data
    images, scores = augment_images(x_test, y_test, dataset=dataset, 
                                    seed_size=seed_size, data_directory_path=data_directory_path, 
                                    categories=categories, augmentation_type=['Inpainting'],
                                    path2graph=path2graph, path2weights=path2weights, 
                                    fidelity_analysis=fidelity_analysis)

    print("size of augmented data set:", len(images))
    if len(scores)!=0:
        with open('./results/' + file_name, 'w') as out_file:
            tsv_writer = csv.writer(out_file, delimiter='\t')
            tsv_writer.writerow(entete_results)
            for l in scores:
                tsv_writer.writerow(l)
    with open('./results/' + file_images, 'w') as out_file:
        tsv_writer = csv.writer(out_file, delimiter='\t')
        tsv_writer.writerow(entete_imgs)
        for l in images:
            tsv_writer.writerow(l)

    print("saved csv")
   
    print("--- %s seconds ---" % (datetime.now() - start_time))


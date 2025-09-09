from augmentation.InpaintingDifussionModel import Inpainting
from process import augment_images
from nlp.Caption_Enrichement_NLP import *

import numpy as np
import torch
from matplotlib import pyplot as plt
# from tensorflow.keras.datasets import cifar10
from utils.datasets import (load_cifar10_dataset, load_custom_dataset, 
                            load_test_imagepaths, load_imagenet_dataset
)
import csv
# import os
import argparse
from datetime import datetime
# import pdb
import yaml


__version__ = 0.2


def parse_arguments():
    """
    Parse command line argument and construct the DNN
    :return: a dictionary comprising the command-line arguments
    """
    text = 'Data Augmentation pipline'
    parser = argparse.ArgumentParser(description=text)
    # new command-line arguments
    parser.add_argument("-V", "--version",  help="show program version",
                        action="version", version="DeepFault %f" % __version__)
    parser.add_argument("-DS", "--dataset", default="cifar10", help="The dataset to be used (imagenet, cifar10, or leaf).", 
                        choices=["imagenet","cifar10","leaf"])
    parser.add_argument("-SS", "--seed_size", default=2, help="size of initial set of seed images.", 
                        type=int)
  
    args = parser.parse_args()

    return vars(args)


def load_config(config_path='config.yaml'):
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    config['iteration'] = int(config['iteration'])
    return config


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

    dataset = args['dataset']
    seed_size = args['seed_size']

    config = load_config(config_path='config.yaml')

    approach = config['methodology'] if config['methodology'] else 'Inpainting'
    approach = 'Inpainting'

    path2graph = config['path2graph']
    path2weights = config['path2weights']

    logfile_name = config['logfile']
    logfile = open(logfile_name, 'a')
    extension='.csv'

    fidelity_analysis = config['fidelity']

    iteration = "_" +str(config['iteration'])+ "_"

    augmented_caption = config['augmented_caption'] if not config['augmented_caption'] == None \
        else 'a person'
    
    device = config['device'] if config['device'] else "cpu"
    print("device:", device)

    # Format as DATE - REGION - REPORT TYPE
    start_time = datetime.now()
    results=[]

    entete_results=['img_id', 'caption', 'augmented_caption','method', 'MSE', 
                    'PSNR', 'FID', 'SSIM', 'LPIPS_alex', 'LPIPS_vgg', 
                    'LPIPS_squeeze', 'VIF']

    # device = torch.device("cpu")
    # entete_imgs=['id', 'caption', 'augmented_caption', 'original_img', 'inpainting', 
    #              'erase', 'noise', 'label', 'augmentation_category']
    entete_imgs=['id', 'caption', 'augmented_caption', 'method', 'original_img', 
                 'augmented_img', 'label', 'augmentation_category']
    

    date = datetime.strftime(datetime.now(), '%Y-%m-%d')
    info = {f'{date} - {approach} - {dataset} - {extension}'}
    extension = '.csv'
    k="scores_matadata"
    a="augmentedset"
    file_name = f'{date}-{approach}-{dataset}-{k}-{iteration}-{extension}'
    file_images=f'{date}-{approach}-{dataset}-{a}-{iteration}-{extension}'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # with torch.cuda.amp.autocast(True):
    ##### dataset and input
    start_time = datetime.now()
    logfile_name = './results/DataAugment.log'
    logfile = open(logfile_name, 'a')

    print(dataset,"loading... ... ...")

    if dataset=='cifar10':
        categories = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog',
                      'frog', 'horse', 'ship', 'truck']
        data_directory_path = "./data/cifar/Augmented/"

        # Load dataset
        (x_train, y_train), (x_test, y_test) = load_cifar10_dataset()
        print("x_test shape:", x_test.shape)
        print("y_test shape:", y_test.shape)
        print(type(x_test))
        print(type(y_test))

    elif dataset=='imagenet':
        categories = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog',
                      'frog', 'horse', 'ship', 'truck']
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
                                    fidelity_analysis=fidelity_analysis, device=device)

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


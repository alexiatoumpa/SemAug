from augmentation.InpaintingDifussionModel import Inpainting
from process import augment_CIFAR_imgs
# from Realism_measures.SSIM import *
# from Realism_measures.FID import *
from nlp.Caption_Enrichement_NLP import *

import numpy as np
import torch
from matplotlib import pyplot as plt
from tensorflow.keras.datasets import cifar10
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
    parser.add_argument("-DS", "--dataset", help="The dataset to be used (mnist\
                        SVHN or cifar10).", choices=["mnist","cifar10","SVHN"])
    parser.add_argument("-Me", "--measure", help="the approach to be employed \
                            to measure similarity", choices=['SSIM', 'FID'])

    parser.add_argument("-Cap", "--Augmented_caption", help="the image caption")
    parser.add_argument("-K", "--iteration", help="nbre of iteration for augmenting an image.", 
                        type=int)
    parser.add_argument("-SS", "--seed_size", help="size of initial set of seed images.", 
                        type=int)
    parser.add_argument("-LOG", "--logfile", help="path to log file")

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
    approach = args['methodology'] if args['methodology'] else 'Inpainting'

    # nlp = spacy.load("en_core_web_sm")
    taxonomy = ['smiling', 'waving', 'talking', 'sleeping', 'siting', 'laughting', 
                'jumping', 'wearing a mask']
    measure = args['measure'] if not args['measure'] == None else 'SSIM'
    # caption = args['intial_caption'] if not args['intial_caption'] == None else 'a person'
    Augmented_caption = args['Augmented_caption'] if not args['Augmented_caption'] == None \
        else 'a person'
    seed_size = args['seed_size'] if not args['seed_size'] == None else 2
    dataset = args['dataset'] if not args['dataset'] == None else 'cifar10'#'coco_animal'#['knw']
    # datatype = args['datatype'] if not args['datatype'] == None else 'cifar'
    logfile_name = args['logfile'] if args['logfile'] else 'DataAugment.log'
    logfile = open(logfile_name, 'a')
    extension='.csv'

    iteration="_11_"
    # Format as DATE - REGION - REPORT TYPE
    start_time = datetime.now()
    results=[]
    # line = [str(id), SSIM_inpaint, SSIM_E, SSIM_N, FID_inpainting, FID_Erase, FID_Noise, clip_score.item()]
    # results.append(line)
    # images.append({"id": str(id), "caption": Initial_caption, "Aug_caption": aug_caption_Category, "original": ini_path,
    #                "Inpainting": inpaint_path, "Erase": erase_path, "Noise": noise_path})

    entete_results=['img_id', 'SSIM_Inp', 'SSIM_Erase','SSIM_Noise', 'FID_inp', 
                    'FID_Erase','FID_Noise', 'Clip score']
    # device = torch.device("cpu")
    entete_imgs=["id", "caption", "Aug_caption", "original_img", "Inpainting", 
                 "Erase", "Noise", "label", "aug_category"]
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
    categories = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 
                  'horse', 'ship', 'truck']
    data_directory_path = "./data/cifar/Augmented/"
    # Load dataset
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    # Create augmented data
    results, images = augment_CIFAR_imgs(np.array([x_test[0]]), np.array([y_test[0]]), 
                                        seed_size=seed_size, data_directory_path=data_directory_path, 
                                        categories=categories)

    print("size of augmented data set:", len(results))
    with open('./results/' + file_name, 'w') as out_file:
        tsv_writer = csv.writer(out_file, delimiter='\t')
        tsv_writer.writerow(entete_results)
        for l in results:
            tsv_writer.writerow(l)
    with open('./results/' + file_images, 'w') as out_file:
        tsv_writer = csv.writer(out_file, delimiter='\t')
        tsv_writer.writerow(entete_imgs)
        for l in images:
            tsv_writer.writerow(l)

    print("saved csv")
   
    print("--- %s seconds ---" % (datetime.now() - start_time))






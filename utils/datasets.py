import os
import random
import cv2
import numpy as np

from tensorflow.keras.datasets import cifar10


def load_cifar10_dataset():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    return (x_train, y_train), (x_test, y_test)

def load_custom_dataset(dataset_path='', image_extensions=None):
    if image_extensions is None:
        image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}

    image_paths = []

    for dirpath, _, filenames in os.walk(dataset_path):
        for file in filenames:
            if os.path.splitext(file)[1].lower() in image_extensions:
                image_paths.append(os.path.join(dirpath, file))

    random.shuffle(image_paths)

    # keep the image labels in the same order as the image paths
    # labels = [os.path.basename(os.path.dirname(path)) for path in image_paths]
    labels = [path.split('/')[-3] for path in image_paths]

    prct = 0.2
    num_train = int(len(image_paths) * (1 - prct))
    x_train_paths = image_paths[:num_train]
    y_train = labels[:num_train]
    x_test_paths = image_paths[num_train:]
    y_test = labels[num_train:]

    x_train_paths = x_train_paths[:1]
    y_train = y_train[:1]
    x_test_paths = x_test_paths[:1]
    y_test = y_test[:1]

    # load images
    x_train = []
    for path in x_train_paths:
        img = cv2.imread(path)
        # resized_img = cv2.resize(img, (32,32))
        try:
            x_train.append(resized_img)
        except Exception:
            x_train.append(img)
    
    x_test = []
    for path in x_test_paths:
        # path = '/home/alexiatoumpa/data/QDC/Grape Varieties_for image processing/Variety4/G4L3E2X7P14/IMG20230720103744.jpg'
        img = cv2.imread(path)
        print("test image path:", path)
        # resized_img = cv2.resize(img, (32,32))
        try:
            x_test.append(resized_img)
        except Exception:
            x_test.append(img)

    # Convert lists to numpy arrays
    x_train = np.array(x_train)
    x_test = np.array(x_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    return (x_train, y_train), (x_test, y_test)


def load_test_images(dataset_path='', image_extensions=None):
    if image_extensions is None:
        image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}

    x_test_paths = []

    for dirpath, _, filenames in os.walk(dataset_path):
        for file in filenames:
            if os.path.splitext(file)[1].lower() in image_extensions:
                x_test_paths.append(os.path.join(dirpath, file))

    random.shuffle(x_test_paths)

    # keep the image labels in the same order as the image paths
    y_test = [path.split('/')[-3] for path in x_test_paths]

    x_test_paths = x_test_paths[:1]
    y_test = y_test[:1]

    # load images  
    x_test = []
    for path in x_test_paths:
        # path = '/home/alexiatoumpa/data/QDC/Grape Varieties_for image processing/Variety4/G4L3E2X7P14/IMG20230720103744.jpg'
        img = cv2.imread(path)
        print("test image path:", path)
        # resized_img = cv2.resize(img, (32,32))
        try:
            x_test.append(resized_img)
        except Exception:
            x_test.append(img)

    # Convert lists to numpy arrays
    x_test = np.array(x_test)
    y_test = np.array(y_test)

    return (_, _), (x_test, y_test)


def load_test_imagepaths(dataset_path='', image_extensions=None):
    if image_extensions is None:
        image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}

    x_test_paths = []

    for dirpath, _, filenames in os.walk(dataset_path):
        for file in filenames:
            if os.path.splitext(file)[1].lower() in image_extensions:
                x_test_paths.append(os.path.join(dirpath, file))

    random.shuffle(x_test_paths)

    # keep the image labels in the same order as the image paths
    y_test = [path.split('/')[-3] for path in x_test_paths]

    return (_, _), (x_test_paths, y_test)



from this import d
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
import shutil
import math
import imutils


SEED_DATA_DIR = "C:\\Users\\deniz\\Downloads\\brain_tumor\\Brain_Data"
num_of_images = {}

for dir in os.listdir(SEED_DATA_DIR):
    num_of_images[dir] = len(os.listdir(os.path.join(SEED_DATA_DIR, dir)))

print(num_of_images)

# lets build 3 folders : 70% train data , 15% validate data , and 15% test

TRAIN_DIR = "C:\\Users\\deniz\\Downloads\\brain_tumor\\Brain_Data\\train"
VALIDATE_DIR = "C:\\Users\\deniz\\Downloads\\brain_tumor\\Brain_Data\\validate"
TEST_DIR = "C:\\Users\\deniz\\Downloads\\brain_tumor\\Brain_Data\\test"

# create the train folder:
if not os.path.exists(TRAIN_DIR):
    os.mkdir(TRAIN_DIR)

    for dir in os.listdir(SEED_DATA_DIR):
        os.makedirs(os.path.join(TRAIN_DIR, dir), exist_ok=True)
        print(os.path.join(TRAIN_DIR, dir))

        images_to_copy = np.random.choice(
            a=os.listdir(os.path.join(SEED_DATA_DIR, dir)),
            size=(max(math.floor(70/100 * num_of_images[dir]) - 5, 1)),
            replace=False
        )

        for img in images_to_copy:
            source_path = os.path.join(SEED_DATA_DIR, dir, img)
            print(source_path)
            dest_path = os.path.join(TRAIN_DIR, dir, img)
            print(dest_path)
            shutil.copy(source_path, dest_path)
            os.remove(source_path)
else:
    print("Train Folder Exists")

# create the test folder
if not os.path.exists(TEST_DIR):
    os.mkdir(TEST_DIR)

    for dir in os.listdir(SEED_DATA_DIR):
        os.makedirs(os.path.join(TEST_DIR, dir), exist_ok=True)
        print(os.path.join(TEST_DIR, dir))

        images_to_copy = np.random.choice(
            a=os.listdir(os.path.join(SEED_DATA_DIR, dir)),
            size=(max(math.floor(15/100 * num_of_images[dir]) - 5, 1)),
            replace=False
        )

        for img in images_to_copy:
            source_path = os.path.join(SEED_DATA_DIR, dir, img)
            print(source_path)
            dest_path = os.path.join(TEST_DIR, dir, img)
            print(dest_path)
            shutil.copy(source_path, dest_path)
            os.remove(source_path)
else:
    print("Test Folder Exists")

# create the validate folder
if not os.path.exists(VALIDATE_DIR):
    os.mkdir(VALIDATE_DIR)

    for dir in os.listdir(SEED_DATA_DIR):
        os.makedirs(os.path.join(VALIDATE_DIR, dir), exist_ok=True)
        print(os.path.join(VALIDATE_DIR, dir))

        images_to_copy = np.random.choice(
            a=os.listdir(os.path.join(SEED_DATA_DIR, dir)),
            size=(max(math.floor(15/100 * num_of_images[dir]) - 5, 1)),
            replace=False
        )

        for img in images_to_copy:
            source_path = os.path.join(SEED_DATA_DIR, dir, img)
            print(source_path)
            dest_path = os.path.join(VALIDATE_DIR, dir, img)
            print(dest_path)
            shutil.copy(source_path, dest_path)
            os.remove(source_path)
else:
    print("Validate Folder Exists")
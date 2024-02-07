import glob
import os
import zipfile

import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from skimage.transform import resize
from sklearn.model_selection import train_test_split


class DatasetLoader:
    def __init__(self):
        self.train_files = None
        self.test_files = None

        self.dataset_archive_path = "assets/kaggle/dogsvscats.zip"
        self.dataset_path = "assets/kaggle/"
        self.train_dir = 'assets/kaggle/train/train'
        self.test_dir = 'assets/kaggle/test1/test1'

        self.dataset_folder = os.listdir(self.dataset_path)
        self.load_images()

    def extract_dataset(self):
        if "test1" not in self.dataset_folder and "train" not in self.dataset_folder:
            try:
                with zipfile.ZipFile(self.dataset_archive_path, "r") as file:
                    file.extractall(self.dataset_path)
            except FileNotFoundError:
                raise FileNotFoundError("Пожалуйста, проверьте, что вы скачали архив с датаетом и он расположен в "
                                        "правильной дерриктории")

    def split_train(self, test_size=0.2):
        train_list, val_list = train_test_split(self.train_files, test_size=test_size)
        return train_list, val_list

    def load_images(self):
        self.train_files = glob.glob(os.path.join(self.train_dir, '*.jpg'))
        self.test_files = glob.glob(os.path.join(self.test_dir, '*.jpg'))
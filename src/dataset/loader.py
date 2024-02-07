# Нужные библиотеки
import glob
import os
import zipfile

from sklearn.model_selection import train_test_split

# Класс для загрузки датасета
class DatasetLoader:
    def __init__(self):
        self.train_files = None
        self.test_files = None
        sep = os.sep
        self.dataset_archive_path = f"assets{sep}kaggle{sep}dogsvscats.zip"
        self.dataset_path = f"assets{sep}kaggle{sep}"
        self.train_dir = f'assets{sep}kaggle{sep}train{sep}train'
        #self.test_dir = f'assets{sep}test1{sep}test1'

        self.dataset_folder = os.listdir(self.dataset_path)
        self.load_images()

    # Разархивация архива
    def extract_dataset(self):
        if "test1" not in self.dataset_folder and "train" not in self.dataset_folder:
            try:
                with zipfile.ZipFile(self.dataset_archive_path, "r") as file:
                    file.extractall(self.dataset_path)
            except FileNotFoundError:
                raise FileNotFoundError("Пожалуйста, проверьте, что вы скачали архив с датаетом и он расположен в "
                                        "правильной дерриктории")

    # Разделение датасета
    def split_train(self, test_size=0.2, val_size=0.4):
        train_list, test_list = train_test_split(self.train_files, test_size=test_size)
        train_list, val_list = train_test_split(train_list, test_size=val_size)
        return train_list, val_list, test_list

    # Загрузка изображений
    def load_images(self):
        self.train_files = glob.glob(os.path.join(self.train_dir, '*.jpg'))
        #self.test_files = glob.glob(os.path.join(self.test_dir, '*.jpg'))
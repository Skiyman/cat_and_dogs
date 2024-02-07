# Нужные библиотеки
import copy
import os
import random

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import wandb
from albumentations.pytorch import ToTensorV2
from matplotlib import pyplot as plt
from torch import optim, nn
from torch.utils.data import DataLoader

from src.dataset.augmentation import transform, val_transforms
from src.dataset.dataset import Dataset
from src.dataset.loader import DatasetLoader
from src.metrics.wandb_metric import init_wandb
from src.model.model import CNN
from src.model.train import ModelTrainer

# Параметры
EPOCH = 15
LEARNING_RATE = 1e-3
BATCH_SIZE = 64
SEED = 42

# Инициализируем seed
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

# Функция для визуализации данных
def display_image_grid(images_filepaths, predicted_labels=None, rows=2, cols=5, name=""):
    figure, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(10, 6))
    random_idx = np.random.randint(1, len(images_filepaths), size=10)
    i = 0
    for idx in random_idx:
        image = cv2.imread(images_filepaths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        true_label = images_filepaths[idx].split(os.sep)[-1].split('.')[0]
        if predicted_labels is not None:
            class_ = predicted_labels[idx] 
            color = "green" if true_label == class_ else "red"
        else:
            class_ = true_label
            color = "green"
        ax.ravel()[i].imshow(image.astype(np.uint8))
        ax.ravel()[i].set_title(class_, color=color)
        ax.ravel()[i].set_axis_off()
        i += 1
    figure.suptitle(name, fontsize=24)
    plt.tight_layout()
    plt.show()

# Функция для визуализации аугментации
def visualize_augmentations(dataset, idx=0, samples=10, cols=5):
    dataset = copy.deepcopy(dataset)
    dataset.transform = A.Compose([t for t in dataset.transform if not isinstance(t, (A.Normalize, ToTensorV2))])
    rows = samples // cols
    figure, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(10, 6))

    for i in range(samples):
        image, _ = dataset[idx]
        #image = image.T
        ax.ravel()[i].imshow(image)
        ax.ravel()[i].set_axis_off()
    figure.suptitle("Пример аугментации", fontsize=24)
    plt.tight_layout()
    plt.show()

# Функция для вывода предсказаний
def display_predict(model, test_loader, test_list, device):
    model = model.eval()
    predicted_labels = []
    with torch.no_grad():
        for data, fileid in test_loader:
            data = data.to(device, non_blocking=True)
            output = model(data)
            predictions = F.softmax(output, dim=1)[:, 1].tolist()
            predicted_labels += list(zip(list(fileid), predictions))


    predicted_labels.sort(key=lambda x: int(x[0]))
    idx = list(map(lambda x: x[0], predicted_labels))
    prob = list(map(lambda x: x[1], predicted_labels))
    submission = pd.DataFrame({'id': idx, 'label': prob})

    preds = pd.DataFrame(columns=["id", "class"])

    for i in range(len(submission)):
        label = submission.label[i]
        if label > 0.5:
            label = 'dog'
        else:
            label = 'cat'
        preds.loc[len(preds.index)] = [submission.id[i], label]

    buff = int(input("Вывести пример работы модели (1-да, 0-нет): "))
    while buff == 1:
        display_image_grid(test_list, preds["class"], name="Пример работы модели")
        buff = int(input("Вывести пример работы модели (1-да, 0-нет): "))


def main():
    # Инициализируем запуск wandb
    wandb.login()
    init_wandb(
        learning_rate=LEARNING_RATE,
        epochs=EPOCH,
        batch_size=BATCH_SIZE
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("device: {0}".format(device))

    # Загрузка датасета
    loader = DatasetLoader()
    loader.extract_dataset()

    seed_everything(SEED)

    # Получение тренировочного, валидационного и тестового наборов данных
    train_list, val_list, test_list = loader.split_train(test_size=0.2, val_size=0.4)
    display_image_grid(train_list, name="Пример данных находящихся в датасете")

    train_data = Dataset(train_list, transform=transform)
    test_data = Dataset(test_list, transform=transform)
    val_data = Dataset(val_list, transform=val_transforms)

    visualize_augmentations(train_data)

    train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(dataset=val_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=True)

    model = CNN().to(device)
    optimizer = optim.Adam(params=model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    print(model)

    # Определение модели для тренировки
    trainer = ModelTrainer(
        device=device,
        train_loader=train_loader,
        val_loader=val_loader,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        epoch=EPOCH
    )
    trainer.start_training()
    torch.save(model.state_dict(), "model.pt")

    display_predict(
        model,
        test_loader,
        test_list,
        device
    )


if __name__ == "__main__":
    main()

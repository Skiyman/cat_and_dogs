import copy
import os
import random

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch import optim, nn
from torch.utils.data import DataLoader

from src.dataset.augmentation import transform, val_transforms
from src.dataset.dataset import Dataset
from src.dataset.loader import DatasetLoader
from src.model.model import CNN
from src.model.train import ModelTrainer

EPOCH = 10
LEARNING_RATE = 1e-3
BATCH_SIZE = 64
SEED = 42


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def display_image_grid(images_filepaths, predicted_labels=pd.DataFrame(), rows=2, cols=5):
    figure, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(20, 12))
    random_idx = np.random.randint(1, len(images_filepaths), size=10)
    i = 0
    for idx in random_idx:
        image = cv2.imread(images_filepaths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        true_label = images_filepaths[idx].split('/')[-1].split('.')[0]
        if predicted_labels.empty:
            class_ = true_label
            color = "green"
        else:
            class_ = predicted_labels.loc[predicted_labels['id'] == true_label, 'class'].values[0]
            color = "red"
        ax.ravel()[i].imshow(image)
        ax.ravel()[i].set_title(class_, color=color)
        ax.ravel()[i].set_axis_off()
        i += 1
    plt.tight_layout()
    plt.show()


def visualize_augmentations(dataset, idx=0, samples=10, cols=5):
    dataset = copy.deepcopy(dataset)
    rows = samples // cols
    figure, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(12, 6))

    for i in range(samples):
        image, _ = dataset[idx]
        image = image.T
        ax.ravel()[i].imshow(image)
        ax.ravel()[i].set_axis_off()
    plt.tight_layout()
    plt.show()


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

    display_image_grid(test_list, preds)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    loader = DatasetLoader()
    loader.extract_dataset()
    seed_everything(SEED)
    train_list, val_list = loader.split_train(test_size=0.2)
    train_data = Dataset(train_list, transform=transform)
    test_data = Dataset(loader.test_files, transform=transform)
    val_data = Dataset(val_list, transform=val_transforms)

    train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(dataset=val_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=True)

    # visualize_augmentations(train_data)

    model = CNN().to(device)
    optimizer = optim.Adam(params=model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    print(model)

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
        loader.test_files,
        device
    )


if __name__ == "__main__":
    main()

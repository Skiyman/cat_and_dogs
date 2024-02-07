import copy
import os
import random

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import optim, nn
from torch.utils.data import DataLoader

from src.dataset.augmentation import transform, val_transforms
from src.dataset.dataset import Dataset
from src.dataset.loader import DatasetLoader
from src.model.model import CNN
from src.model.train import ModelTrainer

EPOCH = 10
LEARNING_RATE = 0.001
BATCH_SIZE = 32
SEED = 42


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


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
    test_loader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(dataset=val_data, batch_size=BATCH_SIZE, shuffle=True)

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


if __name__ == "__main__":
    main()

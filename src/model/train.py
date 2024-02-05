import pandas as pd
import torch
from matplotlib import pyplot as plt
from torch import nn
from tqdm import tqdm

from src.metrics.accuracy_metrics import calculate_accuracy, model_f1_score
from src.metrics.metrics_monitor import MetricMonitor


class ModelTrainer(nn.Module):
    def __init__(self, device, train_loader, val_loader, model, criterion, optimizer, epoch, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.history = pd.DataFrame(columns=['EPOCHS', 'Loss', 'Accuracy', 'Val_loss', 'Val_Accuracy'])
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.epoch = epoch

    def train_model(self, epoch):
        accuracy = None
        loss = None

        metric_monitor = MetricMonitor()
        self.model.train()
        stream = tqdm(self.train_loader)

        for i, (images, target) in enumerate(stream, start=1):
            images = images.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True)
            output = self.model(images)
            loss = self.criterion(output, target)

            accuracy = calculate_accuracy(output, target)
            metric_monitor.update("Loss", loss.item())
            metric_monitor.update("Accuracy", accuracy)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            stream.set_description(
                "Epoch: {epoch}. Train.      {metric_monitor}".format(epoch=epoch, metric_monitor=metric_monitor)
            )

        self.model.eval()
        stream = tqdm(self.val_loader)

        with torch.no_grad():
            for i, (images, target) in enumerate(stream, start=1):
                images = images.to(self.device, non_blocking=True)
                target = target.to(self.device, non_blocking=True)
                output = self.model(images)
                val_loss = self.criterion(output, target)
                val_accuracy = calculate_accuracy(output, target)
                f1 = model_f1_score(output, target)
                metric_monitor.update("F1-score", f1)
                metric_monitor.update("Loss", val_loss.item())
                metric_monitor.update("Accuracy", val_accuracy)
                stream.set_description(
                    "Validation. {metric_monitor}".format(metric_monitor=metric_monitor)
                )

        self.history.loc[len(self.history.index)] = [self, loss.item(), round(accuracy.item() * 100, 2),
                                                     val_loss.item(), round(val_accuracy.item() * 100, 2)]

    def start_training(self):
        for epoch in range(1, self.epoch + 1):
            self.train_model(epoch)
            torch.save(self.model.state_dict(), "model.pt")


    # функции для визуализации результатов
    # TODO: Надо эту функцию адаптировать под wandb
    def visualization(self):
        columns = self.history.columns
        size = columns.size
        colors = ['red', 'orange', 'purple', 'green', 'blue']
        f, axs = plt.subplots(1, size - 1, figsize=(20, 5))

        for i in range(size - 1):
            axs[i].plot(self.history.EPOCHS, self.history[columns[i + 1]],
                        label=columns[i + 1], color=colors[i])
            axs[i].set_xlabel('Эпоха обучения')
            axs[i].set_ylabel(columns[i + 1])
            axs[i].legend()

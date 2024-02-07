import torch
import wandb
from torch import nn
from tqdm import tqdm

from src.metrics.accuracy_metrics import calculate_accuracy, model_f1_score
from src.metrics.metrics_monitor import MetricMonitor


# Класс для обучения
class ModelTrainer(nn.Module):
    def __init__(self, device, train_loader, val_loader, model, criterion, optimizer, epoch, *args, **kwargs):
        super().__init__(*args, **kwargs)
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
        f1 = None

        metric_monitor = MetricMonitor()
        self.model.train()
        stream = tqdm(self.train_loader)

        for i, (images, target) in enumerate(stream, start=1):
            images = images.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True)
            output = self.model(images)
            loss = self.criterion(output, target)
            accuracy = calculate_accuracy(output, target)
            f1 = model_f1_score(output, target)

            metric_monitor.update("Accuracy", accuracy)
            metric_monitor.update("Loss", loss.item())
            metric_monitor.update("F1-score", f1)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            stream.set_description(
                "Epoch: {epoch}.  Train.      {metric_monitor}".format(epoch=epoch, metric_monitor=metric_monitor)
            )

        # Сохранение результатов
        wandb.log({"accuracy": metric_monitor.metrics['Accuracy']['avg'],
                   "loss": metric_monitor.metrics['Loss']['avg'],
                   "F1-score": metric_monitor.metrics['F1-score']['avg']})
        val_accuracy = None
        val_loss = None
        val_f1 = None
        self.model.eval()
        stream = tqdm(self.val_loader)

        with torch.no_grad():
            for i, (images, target) in enumerate(stream, start=1):
                images = images.to(self.device, non_blocking=True)
                target = target.to(self.device, non_blocking=True)
                output = self.model(images)
                val_loss = self.criterion(output, target)
                val_accuracy = calculate_accuracy(output, target)
                val_f1 = model_f1_score(output, target)
                metric_monitor.update("Accuracy", val_accuracy)
                metric_monitor.update("Loss", val_loss.item())
                metric_monitor.update("F1-score", val_f1)
                stream.set_description(
                    "Epoch: {epoch}.  Validation. {metric_monitor}".format(epoch=epoch, metric_monitor=metric_monitor)
                )

        # Сохранение результатов
        wandb.log({"val_accuracy": metric_monitor.metrics['Accuracy']['avg'],
                   "val_loss": metric_monitor.metrics['Loss']['avg'],
                   "val_F1-score": metric_monitor.metrics['F1-score']['avg']})

    def start_training(self):
        for epoch in range(1, self.epoch + 1):
            self.train_model(epoch)
            torch.save(self.model.state_dict(), "model.pt")

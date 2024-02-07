import os

import pandas as pd
import torch
from matplotlib import pyplot as plt
from torch import nn
from tqdm import tqdm

from src.metrics.accuracy_metrics import calculate_accuracy, model_f1_score, model_matrix
from src.metrics.metrics_monitor import MetricMonitor
from src.metrics.wandb_metric import init_wandb

import wandb
from sklearn.metrics import ConfusionMatrixDisplay


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

    def train_model(self):
        accuracy = None
        loss = None

        metric_monitor = MetricMonitor()
        self.model.train()
        stream = tqdm(self.train_loader)
        run = init_wandb()

        

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
                "Train.      {metric_monitor}".format(epoch=self, metric_monitor=metric_monitor)
            )
            
            
        
        wandb.log({'training loss': metric_monitor.metrics['Loss']['avg'], 
            'training accuracy': metric_monitor.metrics['Accuracy']['avg']
             })
        self.model.eval()
        stream = tqdm(self.val_loader)
        metric_monitor.reset()
        

        with torch.no_grad():
            for i, (images, target) in enumerate(stream, start=1):
                images = images.to(self.device, non_blocking=True)
                target = target.to(self.device, non_blocking=True)
                output = self.model(images)
                val_loss = self.criterion(output, target)
                val_accuracy = calculate_accuracy(output, target)
                
                f1 = model_f1_score(output, target)
                conf_matrix = model_matrix(output, target)
                
                
                metric_monitor.update("F1-score", f1)
                metric_monitor.update("Loss", val_loss.item())
                metric_monitor.update("Accuracy", val_accuracy)
                stream.set_description(
                    "Validation. {metric_monitor}".format(metric_monitor=metric_monitor)
                )

        self.history.loc[len(self.history.index)] = [self, loss.item(), round(accuracy.item() * 100, 2),
                                                     val_loss.item(), round(val_accuracy.item() * 100, 2)]
        
        wandb.log({'validation loss': metric_monitor.metrics['Loss']['avg'], 
            'validation accuracy': metric_monitor.metrics['Accuracy']['avg'],
             'f1': metric_monitor.metrics['F1-score']['avg'])



    def start_training(self):
        for epoch in range(1, self.epoch + 1):
            self.train_model()


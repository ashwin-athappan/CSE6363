# # Deep Learning
#
# This assignment covers image classification with deep learning. You will train and evaluate three different categories of networks on the Imagenette dataset. For each model, it is recommended that you implement them using [PyTorch Lightning](http://www.pytorchlightning.ai). Feel free to adapt the code from the [demos we did in class](https://github.com/ajdillhoff/CSE6363/tree/main/deep_learning). A getting started codebase has been provided in this repository.
#
# For all tasks, free CPU compute time and possible GPU time is available through [Google Colab](https://colab.research.google.com/).
#
# # A Basic CNN
#
# The first network will be a basic CNN. This network should include some number of convolutional layers followed by fully connected layers. There is no size requirement for this network nor is there a performance requirement. Train the model until convergence. Implement some form of early stopping in case the model begins to overfit.
#
# In your report, describe the chosen architecture and include the training loss, validation loss, and final test accuracy of the model.
#
# # ResNet 18
#
# The second network will be a ResNet 18. This network is a popular choice for image classification tasks. Train the model until convergence. Implement some form of early stopping in case the model begins to overfit.
#
# In your report, describe the chosen architecture and include the training loss, validation loss, and final test accuracy of the model.
#
# # Regularization
#
# Pick one of the models used in the previous two sections and add regularization in the form of data augmentation or dropout. Train the model until convergence.
#
# In your report, describe your choice of data augmentation and provide a clear comparison of the model with and without regularization.
#
# # Transfer Learning
#
# Transfer learning is an effective way to leverage features learned from another task into a new task. For this part, use a model that was trained on the Imagenette dataset and fine-tune it using the CIFAR10 dataset. You can refer to the class demonstration of [transfer learning](https://github.com/ajdillhoff/CSE6363/blob/main/deep_learning/transfer_learning.ipynb) to help get started.
#
# Using a model from a previous run, re-train it from scratch on the CIFAR10 dataset. Take the same model and initialize it with pre-trained weights from the Imagenette dataset. With the pre-trained model, fine-tune it on the CIFAR10 dataset.
#
# In your report, describe the pre-trained model you chose to use and include the fine-tuning training plots along with the final model accuracy.

import ssl
import torch
import torchmetrics
import numpy as np
import pandas as pd
from torch import nn
import lightning as L
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import Imagenette
from lightning.pytorch import loggers as pl_loggers
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

class BasicCNN(L.LightningModule):
    def __init__(self, num_classes=10):
        super().__init__()

        # Convolutional Layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Fully Connected Layers
        self.fc_layers = nn.Sequential(
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.train_losses = []
        self.val_losses = []

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.train_losses.append(loss.item())
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)

        self.accuracy(y_hat, y)
        self.val_losses.append(loss.item())
        self.log("val_loss", loss)
        self.log("val_accuracy", self.accuracy)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)

        self.accuracy(y_hat, y)

        self.log("test_loss", loss)
        self.log("test_accuracy", self.accuracy)

    def train_epoch(self):
        avg_train_loss = sum(self.train_losses)/len(self.train_losses) if self.train_losses else None
        print(f'Epoch {self.current_epoch}: Average Training Loss:{avg_train_loss}')

    def validation_epoch(self):
        avg_val_loss = sum(self.val_losses)/len(self.val_losses) if self.val_losses else None
        print(f'Epoch {self.current_epoch}: Average Validation Loss:{avg_val_loss}')

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


class RegularizationCNN(L.LightningModule):
    def __init__(self, num_classes=10):
        super().__init__()

        # Convolutional Layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Fully Connected Layers
        self.fc_layers = nn.Sequential(
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(),
            nn.Dropout(0.5),  # Dropout for regularization
            nn.Linear(256, num_classes)
        )

        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.train_losses = []
        self.val_losses = []

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.train_losses.append(loss.item())
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)

        self.accuracy(y_hat, y)
        self.val_losses.append(loss.item())
        self.log("val_loss", loss)
        self.log("val_accuracy", self.accuracy)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)

        self.accuracy(y_hat, y)

        self.log("test_loss", loss)
        self.log("test_accuracy", self.accuracy)

    def train_epoch(self):
        avg_train_loss = sum(self.train_losses)/len(self.train_losses) if self.train_losses else None
        print(f'Epoch {self.current_epoch}: Average Training Loss:{avg_train_loss}')

    def validation_epoch(self):
        avg_val_loss = sum(self.val_losses)/len(self.val_losses) if self.val_losses else None
        print(f'Epoch {self.current_epoch}: Average Validation Loss:{avg_val_loss}')

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

## **Transfer Learning**
class BasicCNN(L.LightningModule):
    def __init__(self, num_classes=10):
        super().__init__()

        # Convolutional Layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Fully Connected Layers
        self.fc_layers = nn.Sequential(
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.train_losses = []
        self.val_losses = []

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.train_losses.append(loss.item())
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)

        self.accuracy(y_hat, y)
        self.val_losses.append(loss.item())
        self.log("val_loss", loss)
        self.log("val_accuracy", self.accuracy)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)

        self.accuracy(y_hat, y)

        self.log("test_loss", loss)
        self.log("test_accuracy", self.accuracy)

    def train_epoch(self):
        avg_train_loss = sum(self.train_losses) / len(self.train_losses) if self.train_losses else None
        print(f'Epoch {self.current_epoch}: Average Training Loss:{avg_train_loss}')

    def validation_epoch(self):
        avg_val_loss = sum(self.val_losses) / len(self.val_losses) if self.val_losses else None
        print(f'Epoch {self.current_epoch}: Average Validation Loss:{avg_val_loss}')

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

# Define BasicCNN and load pre-trained weights
class BasicCNN(L.LightningModule):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )
        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.train_losses = []
        self.val_losses = []

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.functional.cross_entropy(y_hat, y)
        self.train_losses.append(loss.item())
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.functional.cross_entropy(y_hat, y)
        self.accuracy(y_hat, y)
        self.val_losses.append(loss.item())
        self.log("val_loss", loss)
        self.log("val_accuracy", self.accuracy)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.functional.cross_entropy(y_hat, y)
        self.accuracy(y_hat, y)
        self.log("test_loss", loss)
        self.log("test_accuracy", self.accuracy)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)


class ResNet18(L.LightningModule):
    def __init__(self, num_classes=10):
        super(ResNet18, self).__init__()

        # Initial convolutional layer
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Convolutional layers without residual connections
        self.layer1 = self._make_layer(64, 64, stride=1)
        self.layer2 = self._make_layer(64, 128, stride=2)
        self.layer3 = self._make_layer(128, 256, stride=2)
        self.layer4 = self._make_layer(256, 512, stride=2)

        # Fully connected layer for classification
        self.fc = nn.Linear(512, num_classes)

        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.train_losses = []
        self.val_losses = []

    def _make_layer(self, in_channels, out_channels, stride):
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.train_losses.append(loss.item())
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.accuracy(y_hat, y)
        self.val_losses.append(loss.item())
        self.log("val_accuracy", self.accuracy, prog_bar=True)
        self.log("val_loss", loss)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.accuracy(y_hat, y)
        self.log("test_accuracy", self.accuracy, prog_bar=True)
        self.log("test_loss", loss)

    def train_epoch(self):
        avg_train_loss = sum(self.train_losses)/len(self.train_losses) if self.train_losses else None
        print(f'Epoch {self.current_epoch}: Average Training Loss:{avg_train_loss}')

    def validation_epoch(self):
        avg_val_loss = sum(self.val_losses)/len(self.val_losses) if self.val_losses else None
        print(f'Epoch {self.current_epoch}: Average Validation Loss:{avg_val_loss}')

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
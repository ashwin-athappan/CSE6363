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

import torch
from torch import nn
import torch.nn.functional as F
import lightning as L
import torchmetrics

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


from torchvision import transforms
from torchvision.datasets import Imagenette
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint

# Prepare the dataset
train_transforms = transforms.Compose([
    transforms.CenterCrop(160),
    transforms.Resize(64),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    transforms.Grayscale()
])

test_transforms = transforms.Compose([
    transforms.CenterCrop(160),
    transforms.Resize(64),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    transforms.Grayscale()
])

train_dataset = Imagenette("data/imagenette/train/", split="train", size="160px", download=True, transform=train_transforms)

train_set_size = int(len(train_dataset) * 0.9)
val_set_size = len(train_dataset) - train_set_size

seed = torch.Generator().manual_seed(42)
train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_set_size, val_set_size], generator=seed)
val_dataset.dataset.transform = test_transforms

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, num_workers=8, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=128, num_workers=8, shuffle=False)

test_dataset = Imagenette("data/imagenette/test/", split="val", size="160px", download=True, transform=test_transforms)

model = BasicCNN()

early_stop_callback = EarlyStopping(monitor="val_loss",
                                    mode="min",
                                    patience=5)

checkpoint_callback = ModelCheckpoint(
    monitor="val_loss",
    mode="min",
    dirpath="./checkpoints",
    filename="model_weights",
    save_top_k=1,
    verbose=True
)

trainer = L.Trainer(callbacks=[early_stop_callback, checkpoint_callback])
trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=256, num_workers=8, shuffle=False)
trainer.test(model=model, dataloaders=test_loader)


# Plot training and validation losses
def plot_losses(model):
    plt.figure(figsize=(10, 5))
    plt.plot(model.train_losses, label="Training Loss")
    plt.plot(model.val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss over Epochs")
    plt.legend()
    plt.show()

plot_losses(model)

from torchvision import transforms
from torchvision.datasets import Imagenette
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint


# Prepare the dataset
train_transforms = transforms.Compose([
    transforms.CenterCrop(160),
    transforms.Resize(64),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    transforms.Grayscale()
])

test_transforms = transforms.Compose([
    transforms.CenterCrop(160),
    transforms.Resize(64),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    transforms.Grayscale()
])

train_dataset = Imagenette("data/imagenette/train/", split="train", size="160px", download=False, transform=train_transforms)

train_set_size = int(len(train_dataset) * 0.9)
val_set_size = len(train_dataset) - train_set_size

seed = torch.Generator().manual_seed(42)
train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_set_size, val_set_size], generator=seed)
val_dataset.dataset.transform = test_transforms

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, num_workers=8, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=128, num_workers=8, shuffle=False)

test_dataset = Imagenette("data/imagenette/test/", split="val", size="160px", download=False, transform=test_transforms)

model = ResNet18()

early_stop_callback = EarlyStopping(monitor="val_loss",
                                    mode="min",
                                    patience=5)

checkpoint_callback = ModelCheckpoint(
    monitor="val_loss",
    mode="min"
)
trainer = L.Trainer(callbacks=[early_stop_callback, checkpoint_callback])
trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=256, num_workers=8, shuffle=False)
trainer.test(model=model, dataloaders=test_loader)
import matplotlib.pyplot as plt

# Function to plot training and validation losses


## **Regularization**
import torch
from torch import nn
import torch.nn.functional as F
import lightning as L
import torchmetrics

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

from torchvision import transforms
from torchvision.datasets import Imagenette
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint

# Prepare the dataset
train_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomResizedCrop(64, scale=(0.8, 1.0)),
    transforms.ToTensor()
])

test_transforms = transforms.Compose([
    transforms.CenterCrop(160),
    transforms.Resize(64),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    transforms.Grayscale()
])

train_dataset = Imagenette("data/imagenette/train/", split="train", size="160px", download=False, transform=train_transforms)

train_set_size = int(len(train_dataset) * 0.9)
val_set_size = len(train_dataset) - train_set_size

seed = torch.Generator().manual_seed(42)
train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_set_size, val_set_size], generator=seed)
val_dataset.dataset.transform = test_transforms

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, num_workers=8, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=128, num_workers=8, shuffle=False)

test_dataset = Imagenette("data/imagenette/test/", split="val", size="160px", download=False, transform=test_transforms)

model = RegularizationCNN()

early_stop_callback = EarlyStopping(monitor="val_loss",
                                    mode="min",
                                    patience=5)

checkpoint_callback = ModelCheckpoint(
    monitor="val_loss",
    mode="min"
)

trainer = L.Trainer(callbacks=[early_stop_callback, checkpoint_callback])
trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=256, num_workers=8, shuffle=False)
trainer.test(model=model, dataloaders=test_loader)

import matplotlib.pyplot as plt


## **Transfer Learning**
import torch
from torch import nn
import torch.nn.functional as F
import lightning as L
import torchmetrics
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, random_split
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint

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

train_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

test_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

train_dataset = CIFAR10(root="data", train=True, download=True, transform=train_transforms)
test_dataset = CIFAR10(root="data", train=False, download=True, transform=test_transforms)

train_set_size = int(len(train_dataset) * 0.9)
val_set_size = len(train_dataset) - train_set_size

seed = torch.Generator().manual_seed(42)
train_dataset, val_dataset = random_split(train_dataset, [train_set_size, val_set_size], generator=seed)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=8)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=8)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=8)

model = BasicCNN(num_classes=10)

early_stop_callback = EarlyStopping(monitor="val_loss", mode="min", patience=5)
checkpoint_callback = ModelCheckpoint(
    monitor="val_loss",
    mode="min",
    dirpath="./checkpoints",
    filename="cifar10_cnn",
    save_top_k=1,
    verbose=True
)

trainer = L.Trainer(
    max_epochs=10,
    accelerator="gpu",
    devices=1,
    callbacks=[early_stop_callback, checkpoint_callback],
    enable_progress_bar=True
)

trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)

trainer.test(model=model, dataloaders=test_loader)
import matplotlib.pyplot as plt


trainer.save_checkpoint("imagenette_pretrained_weights.ckpt")

import torch
from torch import nn
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, random_split
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint
import lightning as L

train_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

test_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

train_dataset = CIFAR10(root="data", train=True, download=True, transform=train_transforms)
test_dataset = CIFAR10(root="data", train=False, download=True, transform=test_transforms)

train_set_size = int(len(train_dataset) * 0.9)
val_set_size = len(train_dataset) - train_set_size

seed = torch.Generator().manual_seed(42)
train_dataset, val_dataset = random_split(train_dataset, [train_set_size, val_set_size], generator=seed)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=8)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=8)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=8)

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

model = BasicCNN(num_classes=10)
checkpoint = torch.load("imagenette_pretrained_weights.ckpt")
model.load_state_dict(checkpoint["state_dict"])

for param in model.conv_layers.parameters():
    param.requires_grad = False  # Freeze convolutional layers

early_stop_callback = EarlyStopping(monitor="val_loss", mode="min", patience=5)
checkpoint_callback = ModelCheckpoint(
    monitor="val_loss",
    mode="min",
    dirpath="./checkpoints",
    filename="cifar10_finetuned",
    save_top_k=1,
    verbose=True
)

# Trainer setup
trainer = L.Trainer(
    max_epochs=10,
    accelerator="gpu",
    devices=1,
    callbacks=[early_stop_callback, checkpoint_callback]
)

trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)

trainer.test(model=model, dataloaders=test_loader)


plot_losses(model)
import torch
from torch import nn
import torch.nn.functional as F
import lightning as L
import torchmetrics


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
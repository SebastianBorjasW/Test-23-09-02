import torch
import torchvision
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid
from torchinfo import summary
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from model_classes.vgg_model_80 import VGG16

def train_model():
    device  = 'cpu'

    # Definir la ruta del dataset
    rootimg = 'img_train'

    img_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=1),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    dataset = ImageFolder(root= rootimg, transform= img_transform)

    # Definir tamaños de los conjuntos de datos
    train_size = int(0.8 * len(dataset))
    test_val_size = len(dataset) - train_size

    train_dataset, test_val_dataset = random_split(dataset, [train_size, test_val_size])

    test_size = int(0.5 * len(test_val_dataset))
    val_size = len(test_val_dataset) - test_size
    test_dataset, val_dataset = random_split(test_val_dataset, [test_size, val_size])

    # Creación de los data loaders
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    print(len(train_loader), len(test_loader), len(val_loader))

    class VGG16(nn.Module):
        def __init__(self):
            super(VGG16,self).__init__()
            self.model = nn.Sequential(
                # Definición del modelo VGG16
                nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(),
                nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(),
                nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(),
                nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(),
                nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Flatten(),
                nn.Linear(in_features=512*7*7, out_features=4096),
                nn.ReLU(),
                nn.Dropout(),
                nn.Linear(in_features=4096, out_features=2048),
                nn.ReLU(),
                nn.Dropout(),
                nn.Linear(in_features=2048, out_features=80)
            )

        def forward(self, x):
            return self.model(x)

    model = VGG16().to(device)
    #summary(model, input_size=(1,3,224,224))

    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    loss_fn = nn.CrossEntropyLoss()

    epochs = 25
    training_loss = []
    validation_loss = []

    #print(f"Total number of images in the dataset: {len(dataset)}")
    #print(f"Number of classes: {len(dataset.classes)}")


    try:
        for epoch in range(epochs):
            #print("hola")
            model.train()
            epoch_train_loss = 0.0
            print(len(train_loader))
            for images, labels in train_loader:
                print(f"Processing batch with {images.size(0)} images")
                images = images.to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(images)
                #print("hola2")
                train_loss = loss_fn(outputs, labels)
                train_loss.backward()
                optimizer.step()
                epoch_train_loss += train_loss.item() * images.size(0)
                

            epoch_train_loss /= len(train_loader.dataset)
            training_loss.append(epoch_train_loss)

            model.eval()
            epoch_val_loss = 0.0
            with torch.no_grad():
                for val_images, val_labels in val_loader:
                    val_images = val_images.to(device)
                    val_labels = val_labels.to(device)
                    val_outputs = model(val_images)
                    val_loss = loss_fn(val_outputs, val_labels)
                    epoch_val_loss += val_loss.item() * val_images.size(0)

            epoch_val_loss /= len(val_loader.dataset)
            validation_loss.append(epoch_val_loss)

            print(f"Epoch: {epoch+1}/{epochs}, Training Loss: {epoch_train_loss:.4f}, Validation Loss: {epoch_val_loss:.4f}")
    except Exception as e:
        print(f"An error occurred: {e}")

    torch.save(model.state_dict(), 'model.pth')

if __name__ == '__main__':
    train_model()

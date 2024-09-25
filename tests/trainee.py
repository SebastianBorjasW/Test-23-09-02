# Importar librerias
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

#Definir la ruta del dataset
rootimg = '../train'

img_transform = transforms.Compose([
    #rotamos las imagenes con una probabilidad de 0.5 para incrementar
    #la cantidad de datos y hacer el modelo mas robusto
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=1),
    
    #redimensionamos las imagenes para que todas tengan el mismo tamaño
    #y se puedan procesar en el modelo
    transforms.Resize((224, 224)),

    #convertimos las imagenes a tensores para que puedan ser procesadas
    #en un rango de [0,1]
    transforms.ToTensor(),

    #normalizamos los tensores para que tengan una media mas cercana a 0
    #y ayudar a la optimizacion(adam) 
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

#creamos el dataset con las transformaciones definidas
dataset = ImageFolder(root= rootimg, transform= img_transform)

#definimos el tamaño del conjunto de entrenamiento (80%)
train_size = int(0.8 * len(dataset))
#definimos el tamaño del conjunto de entrenamiento (20%)
test_val_size = len(dataset) - train_size

#dividimos el dataset en conjunto de entrenamiento y conjunto de validacion y pueba
train_dataset, test_val_dataset = random_split(dataset, [train_size, test_val_size])

#dividimos el conjunto de validacion y prueba en conjunto de validacion y conjunto de prueba
#10% y 10% del conjunto original para cada uno
test_size = int(0.5 * len(test_val_dataset))
val_size = len(test_val_dataset) - test_size
test_dataset, val_dataset = random_split(test_val_dataset, [test_size, val_size])

# Creacion de data loaders

#definimos el tamaño del lote
batch_size = 30

#creamos los data loaders para cada conjunto de datos
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

print(dataset.classes)

# create vgg16 model
class VGG16(nn.Module):
  def __init__(self):
    super(VGG16,self).__init__()
    self.model = nn.Sequential(

        # first block
        nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),

        # second block
        nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),

        # third block
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

        # fourth block
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

        #fifth block
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

        # fully connected layers
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
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
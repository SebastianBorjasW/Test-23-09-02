# Import Library
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

device = "cuda" if torch.cuda.is_available() else "cpu"

#print(device)

manual_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


dataset = ImageFolder(root='../raw-img', transform=manual_transform)


weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT # .DEFAULT = best available weights 
model = torchvision.models.efficientnet_b0(weights=weights).to(device)

auto_transforms = weights.transforms()
#print(auto_transforms)





train_size = int(0.8 * len(dataset))
test_val_size = len(dataset) - train_size
train_dataset, test_val_dataset = random_split(dataset, [train_size, test_val_size])

# create test and validation set frm test_val_dataset
test_size = int(0.5 * len(test_val_dataset))
val_size = len(test_val_dataset) - test_size
test_dataset, val_dataset = random_split(test_val_dataset, [test_size, val_size])

# check the len of the train, test, val dataset
len(train_dataset), len(test_dataset), len(val_dataset)







# create data loaders
BATCH_SIZE = 32
train_loader = DataLoader(train_dataset,
                          batch_size=BATCH_SIZE,
                          shuffle=True,
                          num_workers=2)

test_loader = DataLoader(test_dataset,
                         batch_size=BATCH_SIZE,
                         shuffle=False,
                         num_workers=2)

val_loader = DataLoader(val_dataset,
                       batch_size=BATCH_SIZE,
                       shuffle=False,
                       num_workers=2)

len(train_loader), len(test_loader), len(val_loader)


class_names = dataset.classes
print(class_names)


torch.manual_seed(42)
fig = plt.figure(figsize=(9, 9))
rows, cols = 4, 4
for i in range(1, rows * cols + 1):
    random_idx = torch.randint(0, len(train_dataset), size=[1]).item()
    img, label = train_dataset[random_idx]
    fig.add_subplot(rows, cols, i)
    plt.imshow(img.permute(2, 1, 0))
    plt.title(class_names[label])
    plt.axis(False)
plt.show()
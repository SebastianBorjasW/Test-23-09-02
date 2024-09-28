import torch
import torchvision
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score

def train_model():
    device = 'cpu'

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

    dataset = ImageFolder(root=rootimg, transform=img_transform)

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
            super(VGG16, self).__init__()
            self.model = nn.Sequential(
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
                nn.Linear(in_features=512 * 7 * 7, out_features=4096),
                nn.ReLU(),
                nn.Dropout(),
                nn.Linear(in_features=4096, out_features=2048),
                nn.ReLU(),
                nn.Dropout(),
                nn.Linear(in_features=2048, out_features=5)
            )

        def forward(self, x):
            return self.model(x)

    model = VGG16().to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    loss_fn = nn.CrossEntropyLoss()

    epochs = 25
    training_loss = []
    validation_loss = []
    training_accuracy = []
    validation_accuracy = []

    try:
        for epoch in range(epochs):
            model.train()
            epoch_train_loss = 0.0
            correct_train = 0
            total_train = 0

            for images, labels in train_loader:
                images = images.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                outputs = model(images)
                train_loss = loss_fn(outputs, labels)
                train_loss.backward()
                optimizer.step()
                epoch_train_loss += train_loss.item() * images.size(0)

                # Calcular precisión en entrenamiento
                _, preds = torch.max(outputs, 1)
                correct_train += (preds == labels).sum().item()
                total_train += labels.size(0)

            epoch_train_loss /= len(train_loader.dataset)
            training_loss.append(epoch_train_loss)
            training_accuracy.append(correct_train / total_train)

            # Validación
            model.eval()
            epoch_val_loss = 0.0
            correct_val = 0
            total_val = 0
            with torch.no_grad():
                for val_images, val_labels in val_loader:
                    val_images = val_images.to(device)
                    val_labels = val_labels.to(device)
                    val_outputs = model(val_images)
                    val_loss = loss_fn(val_outputs, val_labels)
                    epoch_val_loss += val_loss.item() * val_images.size(0)

                    # Calcular precisión en validación
                    _, val_preds = torch.max(val_outputs, 1)
                    correct_val += (val_preds == val_labels).sum().item()
                    total_val += val_labels.size(0)

            epoch_val_loss /= len(val_loader.dataset)
            validation_loss.append(epoch_val_loss)
            validation_accuracy.append(correct_val / total_val)

            print(f"Epoch: {epoch+1}/{epochs}, Training Loss: {epoch_train_loss:.4f}, Validation Loss: {epoch_val_loss:.4f}, "
                  f"Training Accuracy: {correct_train / total_train:.4f}, Validation Accuracy: {correct_val / total_val:.4f}")

    except Exception as e:
        print(f"An error occurred: {e}")

    torch.save(model.state_dict(), 'model2.pth')

    # Graficar curvas de precisión y pérdida
    plt.figure(figsize=(14, 6))

    # Curva de pérdida
    plt.subplot(1, 2, 1)
    plt.plot(range(epochs), training_loss, label='Training Loss')
    plt.plot(range(epochs), validation_loss, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig('loss_curve.png')  # Guardar la gráfica de pérdida

    # Curva de precisión
    plt.subplot(1, 2, 2)
    plt.plot(range(epochs), training_accuracy, label='Training Accuracy')
    plt.plot(range(epochs), validation_accuracy, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.savefig('accuracy_curve.png')  # Guardar la gráfica de precisión

    # Evaluar el modelo en el conjunto de prueba
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for test_images, test_labels in test_loader:
            test_images = test_images.to(device)
            test_labels = test_labels.to(device)
            outputs = model(test_images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(test_labels.cpu().numpy())

    # Calcular precisión en el conjunto de prueba
    test_accuracy = accuracy_score(all_labels, all_preds)
    print(f'Precisión en el conjunto de prueba: {test_accuracy * 100:.2f}%')

    # Matriz de confusión
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=dataset.classes, yticklabels=dataset.classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Matriz de Confusión')
    plt.savefig('confusion_matrix.png')  # Guardar la matriz de confusión

if __name__ == '__main__':
    train_model()

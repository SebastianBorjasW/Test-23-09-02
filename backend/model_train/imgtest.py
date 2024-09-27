import sys
import os
import torch
from torchvision import transforms
from PIL import Image
from model_classes.vgg_model_80 import VGG16
from torch.utils.data import DataLoader, Dataset, random_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np

# Clases para las predicciones
clases = {
    0: 'apple',
    1: 'banana',
    2: 'grape',
    3: 'mango',
    4: 'strawberry'
}

# Cargar el modelo en el dispositivo
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = VGG16().to(device)

try:
    model.load_state_dict(torch.load('../models/model.pth', map_location=device))
    model.eval()  # Establecer el modelo en modo de evaluación
    print("Modelo cargado correctamente.")
except FileNotFoundError as e:
    print(f"Error: No se pudo cargar el modelo. {e}")
    sys.exit(1)
except Exception as e:
    print(f"Ocurrió un error al cargar el modelo: {e}")
    sys.exit(1)

# Transformación para las imágenes
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Definir un Dataset personalizado para cargar las imágenes de entrenamiento y prueba
class ImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        # Recorrer las subcarpetas dentro de image_dir
        for class_label, class_name in clases.items():
            class_folder = os.path.join(image_dir, class_name)
            if os.path.exists(class_folder):
                for img_name in os.listdir(class_folder):
                    img_path = os.path.join(class_folder, img_name)
                    self.image_paths.append(img_path)
                    self.labels.append(class_label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        img = Image.open(img_path).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
        
        return img, label

# Cargar el dataset de entrenamiento y validación
train_image_dir = 'img_train'  # Carpeta con las imágenes de entrenamiento
dataset = ImageDataset(train_image_dir, transform=transform)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Función para entrenar el modelo y guardar métricas
def train_model(model, train_loader, val_loader, optimizer, loss_fn, epochs=25):
    training_loss = []
    validation_loss = []
    training_accuracy = []
    validation_accuracy = []

    for epoch in range(epochs):
        model.train()
        running_train_loss = 0.0
        correct_train = 0
        total_train = 0

        print("hola")

        # Entrenamiento
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)



            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item() * images.size(0)

            _, preds = torch.max(outputs, 1)
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)

        epoch_train_loss = running_train_loss / len(train_loader.dataset)
        epoch_train_acc = correct_train / total_train
        training_loss.append(epoch_train_loss)
        training_accuracy.append(epoch_train_acc)

        # Validación
        model.eval()
        running_val_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for val_images, val_labels in val_loader:
                val_images = val_images.to(device)
                val_labels = val_labels.to(device)

                val_outputs = model(val_images)
                val_loss = loss_fn(val_outputs, val_labels)

                running_val_loss += val_loss.item() * val_images.size(0)
                _, val_preds = torch.max(val_outputs, 1)
                correct_val += (val_preds == val_labels).sum().item()
                total_val += val_labels.size(0)

        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        epoch_val_acc = correct_val / total_val
        validation_loss.append(epoch_val_loss)
        validation_accuracy.append(epoch_val_acc)

        print(f'Epoch [{epoch+1}/{epochs}], '
              f'Train Loss: {epoch_train_loss:.4f}, '
              f'Val Loss: {epoch_val_loss:.4f}, '
              f'Train Acc: {epoch_train_acc:.4f}, '
              f'Val Acc: {epoch_val_acc:.4f}')

    return training_loss, validation_loss, training_accuracy, validation_accuracy

# Configuración del optimizador y la función de pérdida
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
loss_fn = torch.nn.CrossEntropyLoss()

# Entrenar el modelo y obtener las métricas
epochs = 25
train_loss, val_loss, train_acc, val_acc = train_model(model, train_loader, val_loader, optimizer, loss_fn, epochs)

# Graficar curvas de precisión y pérdida
plt.figure(figsize=(14, 6))

# Curva de pérdida
plt.subplot(1, 2, 1)
plt.plot(range(epochs), train_loss, label='Training Loss')
plt.plot(range(epochs), val_loss, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Curva de Pérdida')
plt.legend()

# Curva de precisión
plt.subplot(1, 2, 2)
plt.plot(range(epochs), train_acc, label='Training Accuracy')
plt.plot(range(epochs), val_acc, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Curva de Precisión')
plt.legend()

plt.tight_layout()
plt.show()

# Cargar el dataset de prueba
test_image_dir = 'img_test'  # Carpeta con las imágenes de prueba
test_dataset = ImageDataset(test_image_dir, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Función para evaluar el modelo
def evaluate_model(model, data_loader):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return all_preds, all_labels

# Evaluar el modelo
predictions, true_labels = evaluate_model(model, test_loader)

# Calcular la precisión
accuracy = accuracy_score(true_labels, predictions)
print(f'Precisión del modelo: {accuracy * 100:.2f}%')

# Matriz de confusión
cm = confusion_matrix(true_labels, predictions)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=list(clases.values()), yticklabels=list(clases.values()))
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Matriz de Confusión')
plt.show()

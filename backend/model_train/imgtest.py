import sys
import os
import torch
from torchvision import transforms
from PIL import Image
from model_classes.vgg_model_80 import VGG16
from torch.utils.data import DataLoader, Dataset
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
device = torch.device('cpu')
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

# Definir un Dataset personalizado para cargar las imágenes de prueba
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

# Cargar el dataset de prueba
test_image_dir = 'img_train'  # Carpeta con las imágenes de prueba
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

# Guardar la imagen
output_image_path = 'confusion_matrix.png'
plt.savefig(output_image_path)

# Mostrar la imagen
plt.show()

print(f'Matriz de confusión guardada como {output_image_path}')

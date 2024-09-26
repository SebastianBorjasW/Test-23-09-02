import sys
import os
import torch
from torchvision import transforms
from PIL import Image
from model_classes.vgg_model_80 import VGG16


# Diccionario de clases
clases = {
    0: 'apple',
    1: 'banana',
    2: 'grape',
    3: 'mango',
    4: 'strawberry'
}

# Instanciar el modelo y cargar los pesos
device = torch.device('cpu')
model = VGG16().to(device)
model.load_state_dict(torch.load('../models/model.pth', map_location=device))
model.eval()  # Establecer el modelo en modo de evaluación

# Transformaciones para la imagen de entrada
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

def classify_image(image_path):
    # Cargar la imagen
    img = Image.open(image_path)
    
    # Aplicar las transformaciones
    img_t = transform(img)
    
    # Añadir una dimensión para el lote (batch size = 1)
    batch_t = torch.unsqueeze(img_t, 0)
    
    # Clasificación
    with torch.no_grad():
        output = model(batch_t)
    
    # Obtener el índice de la clase con la probabilidad más alta
    _, predicted = torch.max(output, 1)
    
    return predicted.item()

# Uso del modelo para clasificar una imagen
image_path = 'img_test/Banana/b7.jpeg'  # Cambia esto por la ruta a tu imagen
predicted_class = classify_image(image_path)

# Mostrar el resultado
print(f'La clase predicha es: {clases[predicted_class]}')

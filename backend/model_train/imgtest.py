import sys
import os
import torch
from torchvision import transforms
from PIL import Image
from model_classes.vgg_model_80 import VGG16

clases = {
    0: 'apple',
    1: 'banana',
    2: 'grape',
    3: 'mango',
    4: 'strawberry'
}

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

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

def classify_image(image_path):
    if not os.path.exists(image_path):
        print(f"Error: La ruta de la imagen {image_path} no existe.")
        return

    try:
        img = Image.open(image_path)
        img = img.convert('RGB') 
    except Exception as e:
        print(f"Ocurrió un error al abrir la imagen: {e}")
        return

    img_t = transform(img)
    
    batch_t = torch.unsqueeze(img_t, 0)
    
    
    try:
        with torch.no_grad():
            output = model(batch_t)
        
        _, predicted = torch.max(output, 1)
        return predicted.item()

    except Exception as e:
        print(f"Ocurrió un error durante la clasificación: {e}")
        return

image_path = 'img_test/Apple/a12.jpeg' 
predicted_class = classify_image(image_path)

if predicted_class is not None:
    print(f'La clase predicha es: {clases.get(predicted_class, "Clase desconocida")}')

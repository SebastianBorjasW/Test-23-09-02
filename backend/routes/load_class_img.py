import os
import shutil
import torch
from torchvision import transforms
from PIL import Image
from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import List
from datetime import datetime
from model_train.model_classes.vgg_model_80 import VGG16

app = APIRouter(prefix="/api/load_img", responses={404: {"description": "Not found"}})

UPLOAD_FOLDER = "uploaded_images"
CLASSIFIED_FOLDER = "classified_images"  # Carpeta donde se guardarán las imágenes clasificadas

# Diccionario de clases (actualízalo según tu modelo)
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
model.load_state_dict(torch.load('models/model.pth', map_location=device))
model.eval()  # Establecer el modelo en modo de evaluación

# Transformaciones para la imagen de entrada
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Función para recrear la carpeta donde se guardarán las imágenes
def recreate_upload_folder():
    if os.path.exists(UPLOAD_FOLDER):
        shutil.rmtree(UPLOAD_FOLDER)
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Función para recrear la carpeta de clasificación
def recreate_classified_folder():
    if os.path.exists(CLASSIFIED_FOLDER):
        shutil.rmtree(CLASSIFIED_FOLDER)
    os.makedirs(CLASSIFIED_FOLDER, exist_ok=True)

# Función para clasificar una imagen
def classify_image(image_path):
    # Cargar la imagen
    img = Image.open(image_path).convert('RGB')  # Asegurar que la imagen sea RGB
    
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

# Función para crear una carpeta si no existe
def create_folder_if_not_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

@app.post("/")
async def upload_and_classify_images(files: List[UploadFile] = File(...)):
    # Limpiar y recrear las carpetas antes de procesar nuevas imágenes
    recreate_upload_folder()  
    recreate_classified_folder()  

    processed_files = []

    for file in files:
        # Verificar si el archivo es una imagen
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail=f"El archivo {file.filename} no es una imagen")

        # Generar un nombre único para cada archivo basado en la hora actual
        #timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        file_extension = os.path.splitext(file.filename)[1]  # Obtener la extensión del archivo
        unique_filename = f"{file.filename}"  # Crear un nombre único usando la hora

        # Definir la ruta completa donde se guardará el archivo temporalmente
        file_location = os.path.join(UPLOAD_FOLDER, unique_filename)

        # Guardar el archivo en el servidor temporalmente
        with open(file_location, "wb") as f:
            f.write(await file.read())

        # Clasificar la imagen
        predicted_class_idx = classify_image(file_location)
        predicted_class = clases[predicted_class_idx]

        # Crear la carpeta de la clase si no existe
        class_folder = os.path.join(CLASSIFIED_FOLDER, predicted_class)
        create_folder_if_not_exists(class_folder)

        # Mover la imagen clasificada a la carpeta correspondiente
        final_location = os.path.join(class_folder, unique_filename)
        shutil.move(file_location, final_location)

        processed_files.append({
            "filename": unique_filename,
            "predicted_class": predicted_class,
            "saved_to": final_location
        })

    return {"message": "Imágenes clasificadas y guardadas correctamente", "processed_files": processed_files}

@app.get("/users")
async def users():
    return "hola"

import os
import shutil
import torch
from torchvision import transforms
from PIL import Image
from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import List
from datetime import datetime
from model_train.model_classes.vgg_model_5 import VGG16
from fastapi.responses import FileResponse

app = APIRouter(prefix="/api/load_img", responses={404: {"description": "Not found"}})

UPLOAD_FOLDER = "uploaded_images"
CLASSIFIED_FOLDER = "classified_images" 
ZIP_FOLDER = "classified_images.zip"

clases = {
    0: 'apple',
    1: 'banana',
    2: 'grape',
    3: 'mango',
    4: 'strawberry'
}

device = torch.device('cpu')
model = VGG16().to(device)
model.load_state_dict(torch.load('models/model2.pth', map_location=device))
model.eval() 


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])


def recreate_upload_folder():
    if os.path.exists(UPLOAD_FOLDER):
        shutil.rmtree(UPLOAD_FOLDER)
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def recreate_classified_folder():
    if os.path.exists(CLASSIFIED_FOLDER):
        shutil.rmtree(CLASSIFIED_FOLDER)
    os.makedirs(CLASSIFIED_FOLDER, exist_ok=True)

def classify_image(image_path):
    
    img = Image.open(image_path).convert('RGB')  
    
    img_t = transform(img)
    
    batch_t = torch.unsqueeze(img_t, 0)
    
    with torch.no_grad():
        output = model(batch_t)
    
    _, predicted = torch.max(output, 1)
    
    return predicted.item()

def create_folder_if_not_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


def zip_classified_folder():
    if not os.path.exists(CLASSIFIED_FOLDER):
        raise HTTPException(status_code=404, detail="La carpeta classified_images no existe")

    shutil.make_archive('classified_images', 'zip', CLASSIFIED_FOLDER)

@app.post("/")
async def upload_and_classify_images(files: List[UploadFile] = File(...)):
    recreate_upload_folder()  
    recreate_classified_folder()  

    processed_files = []

    for file in files:
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail=f"El archivo {file.filename} no es una imagen")

        #timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        #file_extension = os.path.splitext(file.filename)[1]
        unique_filename = f"{file.filename}" 

        file_location = os.path.join(UPLOAD_FOLDER, unique_filename)

        with open(file_location, "wb") as f:
            f.write(await file.read())

        predicted_class_idx = classify_image(file_location)
        predicted_class = clases[predicted_class_idx]

        class_folder = os.path.join(CLASSIFIED_FOLDER, predicted_class)
        create_folder_if_not_exists(class_folder)

        final_location = os.path.join(class_folder, unique_filename)
        shutil.move(file_location, final_location)

        processed_files.append({
            "filename": unique_filename,
            "predicted_class": predicted_class,
            "saved_to": final_location
        })

    return {"message": "Imágenes clasificadas y guardadas correctamente", "processed_files": processed_files}

@app.get("/download")
async def download_classified_images():
    if os.path.exists(ZIP_FOLDER):
        os.remove(ZIP_FOLDER)

    zip_classified_folder()

    if not os.path.exists(ZIP_FOLDER):
        raise HTTPException(status_code=500, detail="Error al crear el archivo zip")
    return FileResponse(ZIP_FOLDER, media_type='application/zip', filename=ZIP_FOLDER)
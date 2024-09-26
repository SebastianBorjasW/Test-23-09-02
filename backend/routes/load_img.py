import os
from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import List

app = APIRouter(prefix="/api/load_img", responses={404: {"description": "Not found"}})

UPLOAD_FOLDER = "uploaded_images"  # Carpeta donde se guardarán las imágenes
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Crear la carpeta si no existe

@app.post("/")
async def upload_images(files: List[UploadFile] = File(...)):
    saved_files = []
    
    for file in files:
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail=f"El archivo {file.filename} no es una imagen")
        
        # Ruta para guardar el archivo
        file_location = os.path.join(UPLOAD_FOLDER, file.filename)
        
        # Guardar el archivo en el servidor
        with open(file_location, "wb") as f:
            f.write(await file.read())
        
        saved_files.append(file.filename)
    
    return {"message": "Imágenes recibidas correctamente", "filenames": saved_files}

@app.get("/")
async def users():
    return "hola"

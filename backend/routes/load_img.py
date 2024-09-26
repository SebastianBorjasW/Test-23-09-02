import os
import shutil
from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import List

app = APIRouter(prefix="/api/load_img", responses={404: {"description": "Not found"}})

UPLOAD_FOLDER = "uploaded_images" 


def recreate_upload_folder():
    
    if os.path.exists(UPLOAD_FOLDER):
        shutil.rmtree(UPLOAD_FOLDER)
    
    
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.post("/")
async def upload_images(files: List[UploadFile] = File(...)):
    recreate_upload_folder()

    saved_files = []
    
    for file in files:
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail=f"El archivo {file.filename} no es una imagen")
        
        
        file_location = os.path.join(UPLOAD_FOLDER, file.filename)
        
    
        with open(file_location, "wb") as f:
            f.write(await file.read())
        
        saved_files.append(file.filename)
    
    return {"message": "Imágenes recibidas correctamente", "filenames": saved_files}

@app.get("/users")
async def users():
    return "hola"


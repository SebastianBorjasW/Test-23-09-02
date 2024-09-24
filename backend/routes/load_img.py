from fastapi import APIRouter, UploadFile, File, HTTPException



app = APIRouter(prefix="/api/load_img", responses={404: {"description": "Not found"}})



# Endpoint para recibir la imagen
@app.post("/")
async def upload_image(file: UploadFile = File(...)):
    # Verifica si el archivo es una imagen (puedes hacerlo usando el content_type del archivo)
    if file.content_type.startswith("image/"):
        # Aquí podrías procesar la imagen, guardarla o hacer cualquier operación que necesites.
        # Para este ejemplo, solo imprimiremos "ok" si se recibe correctamente.
        print("ok")
        return {"message": "Imagen recibida correctamente", "filename": file.filename}
    else:
        # En caso de que no sea una imagen, lanzamos una excepción HTTP
        raise HTTPException(status_code=400, detail="El archivo subido no es una imagen")

# Ruta GET para probar que el servidor está activo
@app.get("/")
async def users():
    return "hola"

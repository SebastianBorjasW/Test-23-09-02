from fastapi import APIRouter, UploadFile, File, HTTPException

app = APIRouter(prefix="/api/load_img", responses={404: {"description": "Not found"}})


@app.post("/")
async def upload_image(file: UploadFile = File(...)):
    if file.content_type.startswith("image/"):
        
        
        print(file.filename)
        return {"message": "Imagen recibida correctamente", "filename": file.filename}
    else:
        raise HTTPException(status_code=400, detail="El archivo subido no es una imagen")

@app.get("/")
async def users():
    return "hola"

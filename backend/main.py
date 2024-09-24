from fastapi import FastAPI
from routes import load_img
import uvicorn


app = FastAPI() 
app.include_router(load_img.app)    

@app.get("/")
async def root():
    return {"message": "Hello fast"}



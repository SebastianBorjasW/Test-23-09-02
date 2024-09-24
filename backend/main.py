from fastapi import FastAPI
from routes import load_img
import uvicorn
from fastapi.middleware.cors import CORSMiddleware



app = FastAPI() 
app.include_router(load_img.app)   



app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permitir todos los dominios. Puedes restringirlo a ["http://localhost:1420"]
    allow_credentials=True,
    allow_methods=["*"],  # Permitir todos los métodos (GET, POST, etc.)
    allow_headers=["*"],  # Permitir todos los headers
)

@app.get("/")
async def root():
    return {"message": "Hello fast"}



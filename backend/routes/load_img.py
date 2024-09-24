from fastapi import APIRouter


app = APIRouter(prefix="/api/load_img", responses={404: {"description": "Not found"}})


@app.get("/")
async def users():
    return "hola"

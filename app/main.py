from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from app.api.v1.endpoints import (
    wrap_car,
    #change_color,
)
import os

app = FastAPI()

# Mount the temp directory as a static files directory
os.makedirs("temp", exist_ok=True)
app.mount("/temp", StaticFiles(directory="temp"), name="temp")

# Include the wrap_car router
app.include_router(wrap_car.router)
# app.include_router(change_color.router)

@app.get("/")
def read_root():
    return {"Hello": "MY_Car_Wrap_with AI Generated Sticker"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
from fastapi import FastAPI
from app.api.v1.endpoints import wrap_car

app = FastAPI()

# Include the wrap_car router
app.include_router(wrap_car.router)
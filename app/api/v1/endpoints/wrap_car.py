from fastapi import APIRouter, UploadFile, File
from app.schemas.image import WrapCarRequest
from app.services.car_wrapping import virtual_car_wrap
import cv2
import numpy as np
from io import BytesIO
import base64

router = APIRouter()

@router.post("/wrap-car/")
async def wrap_car(
    car_image: UploadFile = File(...),
    texture_image: UploadFile = File(...),
    brightness: int = 0,
    contrast: float = 1.0
):
    # Read images from file uploads
    car_image_bytes = await car_image.read()
    texture_image_bytes = await texture_image.read()

    car_image = cv2.imdecode(np.frombuffer(car_image_bytes, np.uint8), cv2.IMREAD_COLOR)
    texture_image = cv2.imdecode(np.frombuffer(texture_image_bytes, np.uint8), cv2.IMREAD_COLOR)

    if car_image is None or texture_image is None:
        return {"error": "Invalid image file(s). Please ensure both images are valid."}

    # Perform the car wrapping logic
    wrapped_image = virtual_car_wrap(car_image, texture_image, brightness, contrast)

    # Convert the image to bytes for sending in response
    _, img_encoded = cv2.imencode('.png', wrapped_image)
    wrapped_image_bytes = img_encoded.tobytes()
    wrapped_image_base64 = base64.b64encode(wrapped_image_bytes).decode('utf-8')

    return {"wrapped_car_image": wrapped_image_base64}

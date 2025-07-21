from fastapi import APIRouter, UploadFile, File
from fastapi.responses import FileResponse
from app.schemas.image import WrapCarRequest
from app.services.car_wrapping import _process_car_image
import cv2
import numpy as np
from io import BytesIO
import os

router = APIRouter()

TEMP_DIR = "temp"
os.makedirs(TEMP_DIR, exist_ok=True)

@router.post("/wrap-car/")
async def wrap_car(
    car_image: UploadFile = File(...),
    texture_image: UploadFile = File(...),
    brightness: int = 0,
    contrast: float = 1.0
):
    # Read the uploaded images
    car_image_data = await car_image.read()
    texture_image_data = await texture_image.read()

    # Convert bytes to numpy arrays
    car_np_arr = np.frombuffer(car_image_data, np.uint8)
    texture_np_arr = np.frombuffer(texture_image_data, np.uint8)

    # Decode the images
    car_img = cv2.imdecode(car_np_arr, cv2.IMREAD_COLOR)
    texture_img = cv2.imdecode(texture_np_arr, cv2.IMREAD_COLOR)

    # Convert from BGR to RGB
    car_img_rgb = cv2.cvtColor(car_img, cv2.COLOR_BGR2RGB)
    texture_img_rgb = cv2.cvtColor(texture_img, cv2.COLOR_BGR2RGB)

    # Process the car image
    result_image = _process_car_image(
        car_img_rgb,
        texture_img_rgb,
        brightness,
        contrast
    )

    # Convert back to BGR for saving
    result_bgr = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)

    # Save the result to a temporary file
    output_path = os.path.join(TEMP_DIR, "wrapped_car.png")
    cv2.imwrite(output_path, result_bgr)

    # Return the result image as a file response
    return FileResponse(
        output_path,
        media_type="image/png",
        filename="wrapped_car.png"
    )

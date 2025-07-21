
from fastapi import APIRouter, UploadFile, File
from fastapi.responses import FileResponse
from app.schemas.image import WrapCarRequest
from app.services.change_color import _process_car_image, _parse_color_string
import cv2
import numpy as np
from io import BytesIO
import os

router = APIRouter()

TEMP_DIR = "temp"
os.makedirs(TEMP_DIR, exist_ok=True)

@router.post("/change-color/")
async def change_color(
    car_image: UploadFile = File(...),
    color: str = '#000000',  # Default black color in hex format
    brightness: int = 0,
    contrast: float = 1.0
):
    # Read the uploaded image
    car_image_data = await car_image.read()

    # Convert bytes to numpy array
    car_np_arr = np.frombuffer(car_image_data, np.uint8)

    # Decode the image
    car_img = cv2.imdecode(car_np_arr, cv2.IMREAD_COLOR)

    # Convert from BGR to RGB
    car_img_rgb = cv2.cvtColor(car_img, cv2.COLOR_BGR2RGB)

    # Parse the color string to RGB tuple
    rgb_color = _parse_color_string(color)
    
    # Create a solid color image of the same size as the car image
    color_img_rgb = np.full_like(car_img_rgb, rgb_color)

    # Process the car image
    result_image = _process_car_image(
        car_img_rgb,
        color_img_rgb,
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

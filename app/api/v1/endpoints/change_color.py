from fastapi import APIRouter, UploadFile, File, HTTPException
from app.services.change_color import _process_car_image, _parse_color_string
from fastapi.responses import FileResponse
from typing import Optional
import base64
from fastapi.responses import JSONResponse
from fastapi import APIRouter, UploadFile, File
from app.services.car_wrapping import _process_car_image
import cv2
import numpy as np
import os

# Define the router first
router = APIRouter()

@router.post("/change-color/")
async def change_color(
    car_image: UploadFile = File(..., description="Image of the car (e.g., side view)."),
    wrap_color: Optional[str] = "#FFFFFF",  # Default to white if no color is provided
    brightness: int = 0,  # Integer value for brightness adjustment
    contrast: float = 1.0  # Float value for contrast adjustment
):
    """
    Endpoint to change the color of a car in an image.
    Expects a car image and a color string (hex or rgba), along with brightness and contrast adjustments.
    Returns the color-changed car image as a base64-encoded PNG image string.
    """
    try:
        # Read the uploaded car image data
        car_image_data = await car_image.read()
        # Convert bytes to a NumPy array for OpenCV
        car_np_arr = np.frombuffer(car_image_data, np.uint8)
        # Decode the image. IMREAD_COLOR ensures it's loaded as a color image.
        car_img = cv2.imdecode(car_np_arr, cv2.IMREAD_COLOR)

        # --- NEW CHECK ADDED HERE ---
        # Check if the image was decoded successfully and has valid dimensions
        if car_img is None or car_img.shape[0] == 0 or car_img.shape[1] == 0:
            raise HTTPException(status_code=400, detail="Could not decode car image or image is empty/corrupted. Please upload a valid image.")
        # --- END NEW CHECK ---

        # Convert car image from BGR (OpenCV default) to RGB (for consistent processing)
        car_img_rgb = cv2.cvtColor(car_img, cv2.COLOR_BGR2RGB)

        # Parse the wrap_color string into an RGB tuple
        wrap_color_rgb_tuple = _parse_color_string(wrap_color)
        
        # Create a solid color NumPy array for the wrap material, matching the car image's dimensions
        wrap_material_rgb = np.full(car_img_rgb.shape, wrap_color_rgb_tuple, dtype=np.uint8)

        # Process the car image using the helper function
        result_image_rgb = _process_car_image(
            car_img_rgb,
            wrap_material_rgb,  # Use the solid color material
            brightness,
            contrast
        )

        # Convert the result image back to BGR for saving (OpenCV's imencode expects BGR)
        result_image_bgr = cv2.cvtColor(result_image_rgb, cv2.COLOR_RGB2BGR)

        # Encode the processed image as PNG
        _, buffer = cv2.imencode('.png', result_image_bgr)
        
        # Convert the image buffer to base64
        image_base64 = base64.b64encode(buffer).decode('utf-8')

        # Return the image as a base64 string in the response body
        return JSONResponse(content={"image_base64": image_base64})

    except HTTPException as e:
        # Re-raise HTTPException directly
        raise e
    except Exception as e:
        # Catch any other unexpected errors and return a 500 Internal Server Error
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {str(e)}")

from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from typing import Optional
import cv2
import numpy as np
import os
import io

# Import the necessary functions from your car_wrapping service
# Make sure _process_car_image and _parse_color_string are available in this module
from app.services.change_color import _process_car_image, _parse_color_string


router = APIRouter()

# Directory to temporarily store processed images
TEMP_DIR = "temp"
os.makedirs(TEMP_DIR, exist_ok=True) # Ensure the directory exists

@router.post("/change-color/")
async def change_color(
    car_image: UploadFile = File(..., description="Image of the car (e.g., side view)."),
    wrap_color: Optional[str] = "#FFFFFF", # Default to white if no color is provided
    brightness: int = 0, # Integer value for brightness adjustment
    contrast: float = 1.0 # Float value for contrast adjustment
):
    """
    Endpoint to change the color of a car in an image.
    Expects a car image and a color string (hex or rgba), along with brightness and contrast adjustments.
    Returns the color-changed car image as a PNG file.
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
        # This will be used as the "texture" for the color change
        wrap_material_rgb = np.full(car_img_rgb.shape, wrap_color_rgb_tuple, dtype=np.uint8)

        # Process the car image using the helper function
        # This function will return the image with the new color or a warning message.
        result_image_rgb = _process_car_image(
            car_img_rgb,
            wrap_material_rgb, # Use the solid color material
            brightness,
            contrast
        )

        # Convert the result image back to BGR for saving (OpenCV's imencode expects BGR)
        result_image_bgr = cv2.cvtColor(result_image_rgb, cv2.COLOR_RGB2BGR)

        # Define the output path for the temporary file
        output_filename = f"color_changed_car_{os.urandom(4).hex()}.png" # Unique filename
        output_path = os.path.join(TEMP_DIR, output_filename)
        
        # Save the processed image to a temporary file
        cv2.imwrite(output_path, result_image_bgr)

        # Return the result image as a file response
        # The client will receive this file, and if it contains the warning text,
        # that text will be visible.
        return FileResponse(
            output_path,
            media_type="image/png",
            filename="color_changed_car.png" # Name for download
        )

    except HTTPException as e:
        # Re-raise HTTPException directly
        raise e
    except Exception as e:
        # Catch any other unexpected errors and return a 500 Internal Server Error
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {str(e)}")


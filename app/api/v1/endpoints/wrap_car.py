# import base64
# from fastapi.responses import JSONResponse
# from fastapi import APIRouter, UploadFile, File
# from app.services.car_wrapping import _process_car_image
# import cv2
# import numpy as np
# import os

# router = APIRouter()

# TEMP_DIR = "temp"
# os.makedirs(TEMP_DIR, exist_ok=True)

# @router.post("/wrap-car/")
# async def wrap_car(
#     car_image: UploadFile = File(...),
#     texture_image: UploadFile = File(...),
#     brightness: int = 0,
#     contrast: float = 1.0
# ):
#     try:
#         # Read the uploaded images
#         car_image_data = await car_image.read()
#         texture_image_data = await texture_image.read()

#         # Convert bytes to numpy arrays
#         car_np_arr = np.frombuffer(car_image_data, np.uint8)
#         texture_np_arr = np.frombuffer(texture_image_data, np.uint8)

#         # Decode the images
#         car_img = cv2.imdecode(car_np_arr, cv2.IMREAD_COLOR)
#         texture_img = cv2.imdecode(texture_np_arr, cv2.IMREAD_COLOR)

#         # Convert from BGR to RGB
#         car_img_rgb = cv2.cvtColor(car_img, cv2.COLOR_BGR2RGB)
#         texture_img_rgb = cv2.cvtColor(texture_img, cv2.COLOR_BGR2RGB)

#         # Process the car image
#         result_image = _process_car_image(
#             car_img_rgb,
#             texture_img_rgb,
#             brightness,
#             contrast
#         )

#         # Convert the processed image back to BGR for saving
#         result_bgr = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)

#         # Encode the processed image as PNG
#         _, buffer = cv2.imencode('.png', result_bgr)

#         # Convert the imaFormge buffer to base64
#         image_base64 = base64.b64encode(buffer).decode('utf-8')

#         # Return the image as a base64 string in the response body
#         return JSONResponse(content={"image_base64": image_base64})

#     except Exception as e:
#         # Handle any errors that occur during the image processing
#         return JSONResponse(status_code=500, content={"detail": f"An internal server error occurred: {str(e)}"})

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
import cv2
import numpy as np
import os
import requests
import base64

from urllib.parse import urlparse
from app.services.car_wrapping import _process_car_image
from app.services.sticker_generate import generate_image_wrap_sticker

router = APIRouter()
TEMP_DIR = "temp"
os.makedirs(TEMP_DIR, exist_ok=True)

def is_valid_url(url: str) -> bool:
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except:
        return False

@router.post("/wrap-car/")
async def wrap_car(
    car_image: UploadFile = File(..., description="Upload the car image"),
    texture_prompt: str = Form(..., description="AI-generated car wrap style"),
    wrap_style: str = Form(..., description="Style of the car wrap, e.g., 'glossy', 'metallic', 'matt','carbon-fiber'"),
    brightness: int = Form(0, description="Adjust brightness, default 0"),
    contrast: float = Form(1.0, description="Adjust contrast, default 1.0")
):
    try:
        car_bytes = await car_image.read()
        car_img_np = np.frombuffer(car_bytes, np.uint8)

        car_img = cv2.imdecode(car_img_np, cv2.IMREAD_COLOR)

        if car_img is None:
            raise HTTPException(status_code=400, detail="Invalid car image.")
        
        car_img_rgb = cv2.cvtColor(car_img, cv2.COLOR_BGR2RGB)

        # --- Generate AI texture ---
        new_string = f"AI-generated car wrap style: {texture_prompt}, wrap style: {wrap_style}"
        url = generate_image_wrap_sticker(new_string, '')
        if not is_valid_url(url):
            raise HTTPException(status_code=400, detail="Invalid AI-generated texture URL.")

        resp = requests.get(url, timeout=10)
        resp.raise_for_status()

        style_path = os.path.join(TEMP_DIR, "texture_image.png")
        with open(style_path, 'wb') as f:
            f.write(resp.content)

        texture_img = cv2.imdecode(np.frombuffer(resp.content, np.uint8), cv2.IMREAD_COLOR)
        if texture_img is None:
            raise HTTPException(status_code=400, detail="Downloaded texture image is invalid.")

        texture_img_rgb = cv2.cvtColor(texture_img, cv2.COLOR_BGR2RGB)

        # --- Apply car wrap ---
        result_img = _process_car_image(car_img_rgb, texture_img_rgb, brightness, contrast)
        result_bgr = cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR)

        # --- Save temporary result ---
        result_path = os.path.join(TEMP_DIR, "wrapped_car.png")
        cv2.imwrite(result_path, result_bgr)

        # --- Encode to base64 for API response ---
        _, buffer = cv2.imencode(".png", result_bgr)
        image_base64 = base64.b64encode(buffer).decode("utf-8")

        return JSONResponse(content={
            "image_base64": image_base64
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image processing failed: {str(e)}")

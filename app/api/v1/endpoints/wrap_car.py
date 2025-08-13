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


from fastapi import APIRouter, UploadFile,HTTPException
from fastapi import File, Form
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import io
import os
import requests
import base64
from typing import Optional
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

from typing import Union

@router.post("/wrap-car/")
async def wrap_car(
    car_image: UploadFile = File(...),
    texture_image: Union[UploadFile, str, None] = File(default=None),
    texture_image_ai_generated_prompt: Optional[str] = Form(None),
    brightness: int = Form(0),
    contrast: float = Form(1.0)
):

    try:
        print("texture_image:", texture_image)

        # --- Read & save car image ---
        car_bytes = await car_image.read()
        car_img_np = np.frombuffer(car_bytes, np.uint8)
        car_img = cv2.imdecode(car_img_np, cv2.IMREAD_COLOR)
        if car_img is None:
            raise HTTPException(status_code=400, detail="Invalid car image.")
        car_img_rgb = cv2.cvtColor(car_img, cv2.COLOR_BGR2RGB)

        # --- Load texture image ---
        texture_img = None
        if texture_image:
            tex_bytes = await texture_image.read()
            texture_img = cv2.imdecode(np.frombuffer(tex_bytes, np.uint8), cv2.IMREAD_COLOR)
            if texture_img is None:
                raise HTTPException(status_code=400, detail="Invalid uploaded texture image.")
        elif texture_image_ai_generated_prompt:
            url = generate_image_wrap_sticker(texture_image_ai_generated_prompt,'')
            print("[INFO] AI-generated texture URL:", url)
            if not is_valid_url(url):
                raise HTTPException(status_code=400, detail="Invalid AI-generated texture URL.")

            resp = requests.get(url, timeout=10)
            resp.raise_for_status()

            texture_img = cv2.imdecode(np.frombuffer(resp.content, np.uint8), cv2.IMREAD_COLOR)
            if texture_img is None:
                raise HTTPException(status_code=400, detail="Downloaded texture image is invalid.")

        if texture_img is None:
            raise HTTPException(status_code=400, detail="No texture provided.")

        texture_img_rgb = cv2.cvtColor(texture_img, cv2.COLOR_BGR2RGB)

        # --- Process car wrap ---
        result_img = _process_car_image(car_img_rgb, texture_img_rgb, brightness, contrast)
        result_bgr = cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR)

        # --- Save temporary result ---
        result_path = os.path.join(TEMP_DIR, "wrapped_car.png")
        cv2.imwrite(result_path, result_bgr)

        # --- Encode base64 ---
        _, buffer = cv2.imencode(".png", result_bgr)
        image_base64 = base64.b64encode(buffer).decode("utf-8")

        # --- Return both formats ---
        return JSONResponse(content={
            "image_base64": image_base64, # you can expose this via a static route
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image processing failed: {str(e)}")

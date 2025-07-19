import cv2
import numpy as np
from ultralytics import YOLO
from app.config import settings

# Load the pre-trained YOLO model
# model_path = 'app/model/best.pt'  # Adjust path accordingly
model_path = settings.MODEL_PATH  # Adjust path accordingly
model = YOLO(model_path)

def virtual_car_wrap(car_image: np.ndarray, texture_image: np.ndarray, brightness: int, contrast: float) -> np.ndarray:
    """
    Wrap the texture image onto the car image and adjust lighting.
    """
    results = model.predict(cv2.cvtColor(car_image, cv2.COLOR_RGB2BGR), conf=0.5, verbose=False)
    result = results[0]

    car_mask_raw = None
    if result.masks is not None:
        for i, class_id in enumerate(result.boxes.cls):
            if model.names[int(class_id)] == 'car':
                car_mask_raw = result.masks[i].data[0].cpu().numpy()
                break

    if car_mask_raw is None:
        return car_image  # No car detected, return original image

    # Apply texture wrapping logic (same as previous)
    car_mask_full_size = cv2.resize(car_mask_raw, (car_image.shape[1], car_image.shape[0]), interpolation=cv2.INTER_NEAREST)
    contours, _ = cv2.findContours(car_mask_full_size.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return car_image

    x, y, w, h = cv2.boundingRect(contours[0])
    src_pts = np.array([[0, 0], [texture_image.shape[1], 0], [texture_image.shape[1], texture_image.shape[0]], [0, texture_image.shape[0]]], dtype=np.float32)
    dst_pts = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.float32)
    
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped_texture = cv2.warpPerspective(texture_image, M, (car_image.shape[1], car_image.shape[0]))

    original_car_area = cv2.bitwise_and(car_image, car_image, mask=car_mask_full_size.astype(np.uint8))
    gray_car = cv2.cvtColor(original_car_area, cv2.COLOR_RGB2GRAY)
    
    lighting_map = cv2.convertScaleAbs(gray_car, alpha=contrast, beta=brightness)
    lighting_map_3_channel = cv2.cvtColor(lighting_map, cv2.COLOR_GRAY2RGB)
    
    def overlay_blend(base, overlay):
        base_f, overlay_f = base.astype(np.float32)/255.0, overlay.astype(np.float32)/255.0
        mask = base_f > 0.5
        res = np.zeros_like(base_f)
        res[~mask] = 2 * base_f[~mask] * overlay_f[~mask]
        res[mask] = 1 - 2 * (1 - base_f[mask]) * (1 - overlay_f[mask])
        return np.clip(res*255, 0, 255).astype(np.uint8)
        
    wrapped_car_lit = overlay_blend(lighting_map_3_channel, warped_texture)
    
    mask_3d = np.stack([car_mask_full_size]*3, axis=-1)
    final_image = np.where(mask_3d, wrapped_car_lit, car_image)
    
    return final_image

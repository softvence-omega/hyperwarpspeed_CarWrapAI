import cv2
import numpy as np
from ultralytics import YOLO, settings as yolo_settings
from app.config import settings

# Configure Ultralytics settings
for key, value in settings.ULTRALYTICS_CONFIG.items():
    yolo_settings.update({key: value})

# Load the pre-trained YOLO model
model = YOLO(settings.MODEL_PATH)

def overlay_blend(base, overlay):
    """
    Blends two RGB images using an overlay-like mode.
    base: The lighting map (RGB, uint8)
    overlay: The warped texture (RGB, uint8)
    """
    # Convert to float32 for calculations
    base_f, overlay_f = base.astype(np.float32) / 255.0, overlay.astype(np.float32) / 255.0

    # Create a mask for pixels brighter than mid-gray (0.5)
    mask = base_f > 0.5

    res = np.zeros_like(base_f)

    # Apply different blend modes based on the mask
    # Darker areas (where base_f <= 0.5): Multiply blend (2 * base * overlay)
    res[~mask] = 2 * base_f[~mask] * overlay_f[~mask]
    # Brighter areas (where base_f > 0.5): Screen blend (1 - 2 * (1 - base) * (1 - overlay))
    res[mask] = 1 - 2 * (1 - base_f[mask]) * (1 - overlay_f[mask])

    # Clip values to 0-1 and convert back to uint8 (0-255)
    return np.clip(res * 255, 0, 255).astype(np.uint8)


# --- Core Car Processing Logic (Shared Helper Function) ---
def _process_car_image(original_image, wrap_material_rgb, brightness, contrast):
    """
    Internal helper function to perform car segmentation, warping, and lighting adjustments.
    original_image: The original car image (RGB NumPy array).
    wrap_material_rgb: The texture image to apply (RGB NumPy array).
    brightness: Brightness adjustment value.
    contrast: Contrast adjustment value.
    """
    # print(f"\n--- _process_car_image called ---")
    # print(f"Original image shape: {original_image.shape}, dtype: {original_image.dtype}")
    # print(f"Wrap material shape: {wrap_material_rgb.shape}, dtype: {wrap_material_rgb.dtype}")

    if model is None:
        # print("Model not loaded. Cannot process car. Returning placeholder.")
        return np.zeros((original_image.shape[0], original_image.shape[1], 3), dtype=np.uint8)

    # Convert original_image (RGB) to BGR for YOLOv8 model prediction
    image_bgr_for_model = cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR)
    # print(f"Image for model prediction (BGR) shape: {image_bgr_for_model.shape}, dtype: {image_bgr_for_model.dtype}")

    # Set confidence threshold for prediction. Adjust 'conf' here for debugging.
    results = model.predict(image_bgr_for_model, conf=0.25, verbose=False)
    result = results[0]

    car_mask_raw = None
    if result.masks is not None and len(result.masks.data) > 0:
        # Iterate through detected objects to find the 'car' class
        for i, class_id_tensor in enumerate(result.boxes.cls):
            class_id = int(class_id_tensor.cpu().numpy())
            if model.names[class_id] == 'car':
                car_mask_raw = result.masks[i].data[0].cpu().numpy()
                # print(f"Car mask detected! Raw mask shape: {car_mask_raw.shape}, dtype: {car_mask_raw.dtype}")
                break

    if car_mask_raw is None:
        # print("No car detected by the model. Returning original image with message.")
        output_with_message = original_image.copy()
        cv2.putText(output_with_message, "No car detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        return output_with_message

    car_mask_full_size = cv2.resize(car_mask_raw,
                                    (original_image.shape[1], original_image.shape[0]),
                                    interpolation=cv2.INTER_NEAREST)
    print(f"Car mask full size shape: {car_mask_full_size.shape}, dtype: {car_mask_full_size.dtype}")

    contours, _ = cv2.findContours(car_mask_full_size.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        # print("No contours found for the car mask. Returning original image.")
        return original_image

    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    # print(f"Car bounding box: x={x}, y={y}, w={w}, h={h}")

    src_pts = np.array([[0, 0], [wrap_material_rgb.shape[1], 0],
                        [wrap_material_rgb.shape[1], wrap_material_rgb.shape[0]], [0, wrap_material_rgb.shape[0]]], dtype=np.float32)
    dst_pts = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.float32)

    M = cv2.getPerspectiveTransform(src_pts, dst_pts)

    # Ensure wrap_material_rgb is BGR for warpPerspective
    wrap_material_bgr = cv2.cvtColor(wrap_material_rgb, cv2.COLOR_RGB2BGR)
    warped_material_bgr = cv2.warpPerspective(wrap_material_bgr, M, (original_image.shape[1], original_image.shape[0]))
    warped_material_rgb = cv2.cvtColor(warped_material_bgr, cv2.COLOR_BGR2RGB)
    # print(f"Warped material (RGB) shape: {warped_material_rgb.shape}")

    original_car_area_rgb = cv2.bitwise_and(original_image, original_image, mask=car_mask_full_size.astype(np.uint8))
    gray_car = cv2.cvtColor(original_car_area_rgb, cv2.COLOR_RGB2GRAY)
    # print(f"Gray car area shape: {gray_car.shape}, min/max: {gray_car.min()}, {gray_car.max()}")

    lighting_map = cv2.convertScaleAbs(gray_car, alpha=contrast, beta=brightness)
    lighting_map_3_channel = cv2.cvtColor(lighting_map, cv2.COLOR_GRAY2RGB)
    # print(f"Lighting map 3-channel (RGB) shape: {lighting_map_3_channel.shape}, min/max: {lighting_map_3_channel.min()}, {lighting_map_3_channel.max()}")

    wrapped_car_lit = overlay_blend(lighting_map_3_channel, warped_material_rgb)
    # print(f"Wrapped car lit shape after blending: {wrapped_car_lit.shape}, min/max: {wrapped_car_lit.min()}, {wrapped_car_lit.max()}")

    mask_3d = np.stack([car_mask_full_size.astype(bool)] * 3, axis=-1)
    final_image = np.where(mask_3d, wrapped_car_lit, original_image)
    # print(f"Final image composed. Shape: {final_image.shape}, min/max: {final_image.min()}, {final_image.max()}")

    return final_image

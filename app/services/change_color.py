import cv2
import re
import numpy as np
from ultralytics import YOLO, settings as yolo_settings
from app.config import settings

# Configure Ultralytics settings
for key, value in settings.ULTRALYTICS_CONFIG.items():
    yolo_settings.update({key: value})

# Load the pre-trained YOLO model
model = YOLO(settings.MODEL_PATH)

def _parse_color_string(color_string):
    """
    Parses a color string (hex or rgba) and returns an RGB tuple (R, G, B).
    Ensures a valid 3-element tuple is always returned.
    """
    if color_string is None:
        return (0, 0, 0) # Default to black

    try:
        if isinstance(color_string, tuple) and len(color_string) == 3:
            # Already an RGB tuple (e.g., if passed directly)
            return color_string
        elif color_string.startswith('#'):
            hex_color = color_string.lstrip('#')
            if len(hex_color) == 6:
                rgb_tuple = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
                return rgb_tuple
            else:
                return (0, 0, 0)
        elif color_string.startswith('rgba'):
            # Regex to capture R, G, B values (can be float or int)
            match = re.match(r'rgba\((\d+\.?\d*),\s*(\d+\.?\d*),\s*(\d+\.?\d*),\s*[\d.]+\)', color_string)
            if match:
                # Convert captured strings to float, then to int (clamping to 0-255)
                r, g, b = map(lambda x: int(float(x)), match.groups())
                rgb_tuple = (r, g, b)
                return rgb_tuple
            else:
                return (0, 0, 0)
        else:
            return (0, 0, 0)
    except Exception as e:
        return (0, 0, 0)

# --- Helper function for overlay blending (shared utility) ---
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


# Add this function before _process_car_image
def is_side_view(w, h, min_ratio=1.8, max_ratio=5.0):
    """
    Determine if the detected car is likely in side view based on aspect ratio.
    
    Args:
        w: Width of the car bounding box
        h: Height of the car bounding box
        min_ratio: Minimum width/height ratio for side view (default: 1.8)
        max_ratio: Maximum width/height ratio for side view (default: 5.0)
        
    Returns:
        bool: True if the car appears to be in side view, False otherwise
    """
    if h == 0:  # Avoid division by zero
        return False
        
    aspect_ratio = w / h
    return min_ratio <= aspect_ratio <= max_ratio

# --- Core Car Processing Logic (Shared Helper Function) ---
def _process_car_image(original_image, wrap_material_rgb, brightness, contrast):
    """
    Internal helper function to perform car segmentation, warping, and lighting adjustments.
    original_image: The original car image (RGB NumPy array).
    wrap_material_rgb: The texture or solid color image to apply (RGB NumPy array).
    brightness: Brightness adjustment value.
    contrast: Contrast adjustment value.
    
    Returns:
        np.ndarray: The processed image (RGB NumPy array) with the wrap applied,
                    or the original image with a warning message if no car is detected
                    or if it's not a side view.
    """
    # Create a copy of the original image to draw messages on if needed
    output_image = original_image.copy()

    if model is None:
        cv2.putText(output_image, "Model not loaded. Cannot process car.", 
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        return output_image # Return only the image

    # Convert original_image (RGB) to BGR for YOLOv8 model prediction
    image_bgr_for_model = cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR)

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
                break

    if car_mask_raw is None:
        cv2.putText(output_image, "No car detected", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        return output_image # Return only the image

    car_mask_full_size = cv2.resize(car_mask_raw,
                                    (original_image.shape[1], original_image.shape[0]),
                                    interpolation=cv2.INTER_NEAREST)

    contours, _ = cv2.findContours(car_mask_full_size.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        cv2.putText(output_image, "Could not segment car clearly", 
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        return output_image # Return only the image

    # After finding the largest contour and calculating the bounding box
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    # Check if the car is in side view
    if not is_side_view(w, h):
        output_with_message = original_image.copy()
        
        # Add text with background rectangle
        text = "Please upload side view of the car"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        font_thickness = 2
        text_color = (255, 255, 255)  # White text
        bg_color = (255, 0, 0) # Red background

        
        # Get text size
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)
        
        # Define rectangle coordinates
        rect_start = (10, 20)
        rect_end = (rect_start[0] + text_width + 20, rect_start[1] + text_height + 20)
        text_pos = (rect_start[0] + 10, rect_start[1] + text_height + 10)
        
        # Draw rectangle and text
        output_with_message=cv2.resize(output_with_message, (612,459))
        cv2.rectangle(output_with_message, rect_start, rect_end, bg_color, -1)  # -1 fills the rectangle
        cv2.putText(output_with_message, text, text_pos, font, font_scale, text_color, font_thickness, cv2.LINE_AA)
        
        return output_with_message
    
    # Continue with the existing code for successful processing
    src_pts = np.array([[0, 0], [wrap_material_rgb.shape[1], 0],
                        [wrap_material_rgb.shape[1], wrap_material_rgb.shape[0]], [0, wrap_material_rgb.shape[0]]], dtype=np.float32)
    dst_pts = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.float32)

    M = cv2.getPerspectiveTransform(src_pts, dst_pts)

    # Ensure wrap_material_rgb is BGR for warpPerspective
    wrap_material_bgr = cv2.cvtColor(wrap_material_rgb, cv2.COLOR_RGB2BGR)
    warped_material_bgr = cv2.warpPerspective(wrap_material_bgr, M, (original_image.shape[1], original_image.shape[0]))
    warped_material_rgb = cv2.cvtColor(warped_material_bgr, cv2.COLOR_BGR2RGB)

    original_car_area_rgb = cv2.bitwise_and(original_image, original_image, mask=car_mask_full_size.astype(np.uint8))
    gray_car = cv2.cvtColor(original_car_area_rgb, cv2.COLOR_RGB2GRAY)

    lighting_map = cv2.convertScaleAbs(gray_car, alpha=contrast, beta=brightness)
    lighting_map_3_channel = cv2.cvtColor(lighting_map, cv2.COLOR_GRAY2RGB)

    wrapped_car_lit = overlay_blend(lighting_map_3_channel, warped_material_rgb)

    mask_3d = np.stack([car_mask_full_size.astype(bool)] * 3, axis=-1)
    final_image = np.where(mask_3d, wrapped_car_lit, original_image)

    return final_image # Return only the image

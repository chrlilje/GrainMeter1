import numpy as np


def adjust_brightness_contrast(image: np.ndarray, brightness: float = 0.0, contrast: float = 1.0) -> np.ndarray:
    # brightness in range [-1.0, 1.0], contrast multiplier
    img = image.astype('float32') * contrast + (brightness * 255.0)
    img = np.clip(img, 0, 255).astype('uint8')
    return img

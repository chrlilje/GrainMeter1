from dataclasses import dataclass


@dataclass
class CalibrationData:
    pixel_distance: float
    known_distance_mm: float
    pixels_per_mm: float
    mm_per_pixel: float
    um_per_pixel: float


def from_pixel_and_real(pixel_distance: float, known_distance_mm: float) -> CalibrationData:
    if pixel_distance <= 0 or known_distance_mm <= 0:
        raise ValueError("Distances must be positive")
    pixels_per_mm = pixel_distance / known_distance_mm
    mm_per_pixel = known_distance_mm / pixel_distance
    um_per_pixel = mm_per_pixel * 1000.0
    return CalibrationData(pixel_distance, known_distance_mm, pixels_per_mm, mm_per_pixel, um_per_pixel)

from dataclasses import dataclass


@dataclass
class CalibrationData:
    pixel_distance: float
    known_distance_mm: float
    pixels_per_mm: float
    mm_per_pixel: float
    um_per_pixel: float

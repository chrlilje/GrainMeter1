from dataclasses import dataclass


@dataclass
class Measurement:
    id: int
    measurement_type: str
    x1: float
    y1: float
    x2: float
    y2: float
    pixel_length: float
    length_mm: float
    length_um: float
    intersect_points: tuple[tuple[float, float], ...] = ()
    grain_size_mm: float | None = None
    grain_size_um: float | None = None
    comment: str = ""
    accepted: bool = True

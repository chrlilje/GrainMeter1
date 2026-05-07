import csv
from typing import Iterable

try:
    from .models.measurement import Measurement as ModelMeasurement
    from .calibration import CalibrationData
except ImportError:
    from models.measurement import Measurement as ModelMeasurement
    from calibration import CalibrationData


def export_measurements_csv(path: str, measurements: Iterable[ModelMeasurement], calibration: CalibrationData | None, overlay_scale: float = 1.0) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        # header with calibration
        writer.writerow(["image_name", "reference_image_name", "known_distance_mm", "pixels_per_mm", "um_per_pixel", "measurement_id", "measurement_type", "x1", "y1", "x2", "y2", "pixel_length", "length_mm", "length_um", "intersection_count", "grain_size_mm", "grain_size_um", "intersection_points", "comment", "accepted"])
        for m in measurements:
            # compute scaled values according to overlay_scale and calibration if available
            pixel_length = m.pixel_length * float(overlay_scale)
            if calibration:
                length_mm = pixel_length * calibration.mm_per_pixel
                length_um = pixel_length * calibration.um_per_pixel
            else:
                # fallback to stored values scaled proportionally
                scale = (pixel_length / m.pixel_length) if m.pixel_length else 1.0
                length_mm = m.length_mm * scale
                length_um = m.length_um * scale

            intersection_points = tuple(getattr(m, "intersect_points", ()) or ())
            intersection_count = len(intersection_points)
            grain_size_mm = getattr(m, "grain_size_mm", None)
            grain_size_um = getattr(m, "grain_size_um", None)
            if m.measurement_type == "intersects" and intersection_count > 0:
                if grain_size_mm is None:
                    grain_size_mm = length_mm / intersection_count
                if grain_size_um is None:
                    grain_size_um = length_um / intersection_count

            writer.writerow([
                "",  # image_name placeholder
                "",  # reference_image_name placeholder
                round(calibration.known_distance_mm, 2) if calibration else "",
                round(calibration.pixels_per_mm, 2) if calibration else "",
                round(calibration.um_per_pixel, 2) if calibration else "",
                m.id,
                m.measurement_type,
                round(m.x1, 2),
                round(m.y1, 2),
                round(m.x2, 2),
                round(m.y2, 2),
                round(pixel_length, 2),
                round(length_mm, 2),
                round(length_um, 2),
                intersection_count,
                round(grain_size_mm, 2) if grain_size_mm is not None else "",
                round(grain_size_um, 2) if grain_size_um is not None else "",
                ";".join(f"{x:.2f}:{y:.2f}" for x, y in intersection_points),
                m.comment,
                m.accepted,
            ])

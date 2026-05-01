from dataclasses import dataclass
from typing import Optional

try:
    from ..calibration import CalibrationData
except ImportError:
    from calibration import CalibrationData


@dataclass
class ProjectState:
    reference_image_path: str | None = None
    sample_image_path: str | None = None
    calibration: Optional[CalibrationData] = None


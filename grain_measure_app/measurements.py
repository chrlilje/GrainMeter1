from dataclasses import dataclass
from typing import List


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


class MeasurementsManager:
    def __init__(self) -> None:
        self._items: List[Measurement] = []
        self._next_id = 1

    def add(self, m: Measurement) -> None:
        m.id = self._next_id
        self._next_id += 1
        self._items.append(m)

    def get_all(self) -> List[Measurement]:
        return list(self._items)

    def count(self) -> int:
        return len(self._items)

    def clear(self) -> None:
        self._items.clear()
        self._next_id = 1

    def remove_by_id(self, measurement_id: int) -> bool:
        for index, item in enumerate(self._items):
            if item.id == measurement_id:
                del self._items[index]
                return True
        return False

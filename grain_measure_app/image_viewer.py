from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from PySide6.QtCore import QPointF, Qt, QRect
from PySide6.QtGui import QColor, QPainter, QPen, QPixmap, QImage
from PySide6.QtWidgets import QWidget
import cv2
import numpy as np


@dataclass
class MeasurementLine:
    x1: float
    y1: float
    x2: float
    y2: float
    measurement_id: int | None = None


class ImageViewer(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)

        self._pixmap: QPixmap | None = None
        self._image_path: str | None = None
        self._image_cache_bgr: np.ndarray | None = None  # Cache original image in BGR
        self._overlay_pixmap: QPixmap | None = None
        self._overlay_path: str | None = None
        self._overlay_cache_bgr: np.ndarray | None = None  # Cache original overlay
        self._overlay_opacity: float = 0.5
        self._overlay_scale: float = 1.0
        self._min_overlay_scale: float = 0.1
        self._max_overlay_scale: float = 10.0
        self._overlay_scale_mode: bool = False
        self._overlay_scale_callback: Optional[Callable[[float], None]] = None
        self._brightness: float = 0.0
        self._contrast: float = 1.0
        self._saturation: float = 1.0
        self._overlay_offset = QPointF(0.0, 0.0)
        self._view_scale: float = 1.0
        self._min_scale: float = 0.05
        self._max_scale: float = 50.0
        self._pan_offset = QPointF(0.0, 0.0)
        self._drag_start_pos: QPointF | None = None
        self._drag_start_pan = QPointF(0.0, 0.0)
        self._overlay_drag_start: QPointF | None = None
        self._overlay_drag_origin = QPointF(0.0, 0.0)

        self._point_selection_enabled = False
        self._point_callback: Optional[Callable[[float, float], None]] = None
        self._hover_callback: Optional[Callable[[float, float], None]] = None
        self._measurements: list[MeasurementLine] = []
        self._selected_measurement_id: int | None = None
        self._temp_line: tuple[float, float, float, float] | None = None
        self._measurement_start: QPointF | None = None
        self._calibration_points: list[QPointF] = []
        self._calibration_preview: tuple[float, float, float, float] | None = None
        self._calibration_label_mm: float | None = None

        # Whether to draw measurement labels (text) on top of measurement lines
        self._show_measurement_labels: bool = False
        # mapping from measurement_id to label text
        self._measurement_label_texts: dict[int, str] = {}

        self._move_overlay_enabled = False
        self._pick_mode = False
        self._view_mode = "navigate"

    def load_image(self, path: str) -> None:
        self._image_path = path
        # Load and cache the original image
        self._image_cache_bgr = self._read_image_to_bgr(path)
        if self._image_cache_bgr is not None:
            self._pixmap = self._enhance_from_cache(self._image_cache_bgr)
        else:
            self._pixmap = QPixmap(path)
        self._reset_view()
        self.update()

    def set_overlay_image(self, path: str | None) -> None:
        self._overlay_path = path
        if not path:
            self._overlay_pixmap = None
            self._overlay_cache_bgr = None
        else:
            # Load and cache the original overlay
            self._overlay_cache_bgr = self._read_image_to_bgr(path)
            if self._overlay_cache_bgr is not None:
                self._overlay_pixmap = self._enhance_from_cache(self._overlay_cache_bgr)
            else:
                self._overlay_pixmap = QPixmap(path)
        self.update()

    def set_enhancement(self, brightness: float = 0.0, contrast: float = 1.0, saturation: float = 1.0) -> None:
        """Apply image enhancements: brightness (-100 to 100), contrast (0.5 to 3.0), saturation (0.0 to 2.0)."""
        self._brightness = float(brightness)
        self._contrast = max(0.5, min(3.0, float(contrast)))
        self._saturation = max(0.0, min(2.0, float(saturation)))
        self._apply_enhancements()
        self.update()

    def set_calibration_label_mm(self, mm: float | None) -> None:
        """Set a label (in mm) to draw over the calibration line if present."""
        try:
            self._calibration_label_mm = float(mm) if mm is not None else None
        except Exception:
            self._calibration_label_mm = None
        self.update()

    def set_show_measurement_labels(self, enabled: bool) -> None:
        self._show_measurement_labels = bool(enabled)
        self.update()

    def set_measurement_label_texts(self, mapping: dict[int, str]) -> None:
        # Store mapping safely (copy)
        try:
            self._measurement_label_texts = dict(mapping) if mapping is not None else {}
        except Exception:
            self._measurement_label_texts = {}
        self.update()

    def _apply_enhancements(self) -> None:
        """Apply enhancements from cached images."""
        if self._image_cache_bgr is not None:
            self._pixmap = self._enhance_from_cache(self._image_cache_bgr)
        if self._overlay_cache_bgr is not None:
            self._overlay_pixmap = self._enhance_from_cache(self._overlay_cache_bgr)

    def _read_image_to_bgr(self, path: str) -> np.ndarray | None:
        """Read image file and return as BGR numpy array, or None if fails."""
        try:
            with open(path, 'rb') as f:
                file_bytes = np.fromfile(f, np.uint8)
            img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            return img_bgr
        except Exception:
            return None

    def _enhance_from_cache(self, img_bgr: np.ndarray) -> QPixmap:
        """Apply enhancements to a cached BGR image and return as QPixmap."""
        try:
            # Convert to float32
            img = img_bgr.astype(np.float32) / 255.0

            # Apply brightness
            img = img + (self._brightness / 100.0)

            # Apply contrast (scale around mid-gray)
            img = (img - 0.5) * self._contrast + 0.5

            # Apply saturation via HSV
            if self._saturation != 1.0:
                img_uint8 = np.clip(img * 255, 0, 255).astype(np.uint8)
                hsv = cv2.cvtColor(img_uint8, cv2.COLOR_BGR2HSV).astype(np.float32)
                hsv[:, :, 1] = np.clip(hsv[:, :, 1] * self._saturation, 0, 255)
                img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32) / 255.0

            # Clamp to valid range and convert back
            img = np.clip(img * 255, 0, 255).astype(np.uint8)

            # Convert BGR to RGB for QImage
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w, ch = img_rgb.shape
            bytes_per_line = ch * w
            qt_img = QImage(img_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()

            return QPixmap.fromImage(qt_img)
        except Exception:
            # Fallback on error
            return QPixmap()

    def set_overlay_scale(self, scale: float) -> None:
        self._overlay_scale = max(self._min_overlay_scale, min(self._max_overlay_scale, float(scale)))
        if self._overlay_scale_callback:
            try:
                self._overlay_scale_callback(self._overlay_scale)
            except Exception:
                pass
        self.update()

    def get_overlay_scale(self) -> float:
        return float(self._overlay_scale)

    def set_overlay_scale_mode(self, enabled: bool) -> None:
        self._overlay_scale_mode = bool(enabled)
        self.setCursor(Qt.SizeHorCursor if enabled else (Qt.CrossCursor if self._pick_mode else Qt.ArrowCursor))

    def connect_overlay_scale_changed(self, callback: Callable[[float], None]) -> None:
        self._overlay_scale_callback = callback

    def disconnect_overlay_scale_changed(self) -> None:
        self._overlay_scale_callback = None

    def set_overlay_opacity(self, opacity: float) -> None:
        self._overlay_opacity = max(0.0, min(1.0, float(opacity)))
        self.update()

    def clear_overlay(self) -> None:
        self._overlay_path = None
        self._overlay_pixmap = None
        self._overlay_offset = QPointF(0.0, 0.0)
        self.update()

    def set_measurements(self, lines: list[MeasurementLine]) -> None:
        self._measurements = list(lines)
        self.update()

    def set_selected_measurement_id(self, measurement_id: int | None) -> None:
        self._selected_measurement_id = measurement_id
        self.update()

    def clear_measurement_preview(self) -> None:
        self._temp_line = None
        self._measurement_start = None
        self.update()

    def set_measurement_preview(self, start: tuple[float, float] | None, end: tuple[float, float] | None) -> None:
        if start is None or end is None:
            self._temp_line = None
        else:
            self._temp_line = (start[0], start[1], end[0], end[1])
        self.update()

    def clear_calibration_marks(self) -> None:
        self._calibration_points = []
        self._calibration_preview = None
        self.update()

    def add_calibration_point(self, point: tuple[float, float]) -> None:
        self._calibration_points.append(QPointF(point[0], point[1]))
        self.update()

    def set_calibration_preview(self, start: tuple[float, float] | None, end: tuple[float, float] | None) -> None:
        if start is None or end is None:
            self._calibration_preview = None
        else:
            self._calibration_preview = (start[0], start[1], end[0], end[1])
        self.update()

    def enable_point_selection(self, enabled: bool) -> None:
        self._point_selection_enabled = enabled
        self._pick_mode = enabled
        if enabled:
            self._view_mode = "pick"
        elif self._view_mode == "pick":
            self._view_mode = "navigate"
        self.setCursor(Qt.CrossCursor if enabled else Qt.ArrowCursor)

    def connect_point(self, callback: Callable[[float, float], None]) -> None:
        self._point_callback = callback

    def disconnect_point(self) -> None:
        self._point_callback = None

    def connect_hover(self, callback: Callable[[float, float], None]) -> None:
        self._hover_callback = callback

    def disconnect_hover(self) -> None:
        self._hover_callback = None

    def set_move_overlay_enabled(self, enabled: bool) -> None:
        self._move_overlay_enabled = enabled

    def set_view_mode(self, mode: str) -> None:
        self._view_mode = mode
        self._pick_mode = mode == "pick"
        self.setCursor(Qt.CrossCursor if self._pick_mode else Qt.ArrowCursor)

    def reset_zoom(self) -> None:
        self._reset_view()
        self.update()

    def _reset_view(self) -> None:
        self._view_scale = 1.0
        self._pan_offset = QPointF(0.0, 0.0)
        self._overlay_offset = QPointF(0.0, 0.0)
        self._measurement_start = None
        self._temp_line = None
        self._calibration_preview = None

    def wheelEvent(self, event) -> None:  # type: ignore[override]
        if not self._pixmap:
            return

        # choose base factors
        base_in = 1.15
        base_out = 1.0 / 1.15

        # microscale when Shift is held: much smaller steps
        modifiers = event.modifiers() if hasattr(event, "modifiers") else Qt.KeyboardModifiers(0)
        if modifiers & Qt.ShiftModifier:
            base_in = 1.01
            base_out = 1.0 / 1.01

        factor = base_in if event.angleDelta().y() > 0 else base_out

        # If overlay-scaling mode is active and we have an overlay, scale the overlay instead
        if self._overlay_scale_mode and self._overlay_pixmap:
            new_overlay_scale = self._overlay_scale * factor
            if new_overlay_scale < self._min_overlay_scale or new_overlay_scale > self._max_overlay_scale:
                return
            self.set_overlay_scale(new_overlay_scale)
            return

        # Default behavior: zoom the whole view. Keep cursor point stable.
        new_scale = self._view_scale * factor
        if new_scale < self._min_scale or new_scale > self._max_scale:
            return

        cursor_pos = QPointF(event.position())

        # Image coordinate under cursor before scaling
        image_point = self.widget_to_image(cursor_pos)

        # Compute base geometry with new scale but current pan offset
        pix_w = self._pixmap.width()
        pix_h = self._pixmap.height()
        base_x_new = (self.width() - pix_w * new_scale) / 2 + self._pan_offset.x()
        base_y_new = (self.height() - pix_h * new_scale) / 2 + self._pan_offset.y()

        # Where that image point would be after scaling (widget coords)
        new_widget_x = base_x_new + image_point.x() * new_scale
        new_widget_y = base_y_new + image_point.y() * new_scale

        # Adjust pan so the image point stays under the cursor
        delta_x = cursor_pos.x() - new_widget_x
        delta_y = cursor_pos.y() - new_widget_y

        self._view_scale = new_scale
        self._pan_offset = QPointF(self._pan_offset.x() + delta_x, self._pan_offset.y() + delta_y)
        self.update()

    def mousePressEvent(self, event) -> None:  # type: ignore[override]
        if not self._pixmap:
            return

        pos = QPointF(event.position())

        if self._pick_mode and event.button() == Qt.LeftButton:
            point = self.widget_to_image(pos)
            if self._point_callback:
                self._point_callback(point.x(), point.y())
            return

        if self._move_overlay_enabled and self._overlay_pixmap and event.button() == Qt.LeftButton:
            self._overlay_drag_start = pos
            self._overlay_drag_origin = QPointF(self._overlay_offset)
            return

        if event.button() == Qt.LeftButton or event.button() == Qt.MiddleButton:
            self._drag_start_pos = pos
            self._drag_start_pan = QPointF(self._pan_offset)

    def mouseMoveEvent(self, event) -> None:  # type: ignore[override]
        if not self._pixmap:
            return

        pos = QPointF(event.position())

        if self._pick_mode and self._hover_callback:
            point = self.widget_to_image(pos)
            self._hover_callback(point.x(), point.y())
            return

        if self._overlay_drag_start is not None and self._overlay_pixmap:
            delta = pos - self._overlay_drag_start
            self._overlay_offset = QPointF(self._overlay_drag_origin.x() + delta.x(), self._overlay_drag_origin.y() + delta.y())
            self.update()
            return

        if self._drag_start_pos is not None:
            delta = pos - self._drag_start_pos
            self._pan_offset = QPointF(self._drag_start_pan.x() + delta.x(), self._drag_start_pan.y() + delta.y())
            self.update()
            return

    def mouseReleaseEvent(self, event) -> None:  # type: ignore[override]
        self._drag_start_pos = None
        self._overlay_drag_start = None
        if self._pick_mode and event.button() == Qt.LeftButton:
            return

    def mouseDoubleClickEvent(self, event) -> None:  # type: ignore[override]
        if event.button() == Qt.LeftButton:
            self.reset_zoom()

    def paintEvent(self, event) -> None:  # type: ignore[override]
        painter = QPainter(self)
        try:
            painter.fillRect(self.rect(), QColor(28, 28, 28))
            painter.setRenderHint(QPainter.SmoothPixmapTransform)

            if not self._pixmap:
                painter.setPen(Qt.white)
                painter.drawText(self.rect(), Qt.AlignCenter, "No image")
                return

            base_width = self._pixmap.width() * self._view_scale
            base_height = self._pixmap.height() * self._view_scale
            base_x = (self.width() - base_width) / 2 + self._pan_offset.x()
            base_y = (self.height() - base_height) / 2 + self._pan_offset.y()
            base_rect = self._scaled_rect(self._pixmap.width(), self._pixmap.height(), base_x, base_y, self._view_scale)
            painter.drawPixmap(base_rect, self._pixmap)

            if self._overlay_pixmap:
                # overlay_scale is applied on top of the view scale
                overlay_rect = self._scaled_rect(self._overlay_pixmap.width(), self._overlay_pixmap.height(), base_x + self._overlay_offset.x(), base_y + self._overlay_offset.y(), self._view_scale * self._overlay_scale)
                painter.setOpacity(self._overlay_opacity)
                painter.drawPixmap(overlay_rect, self._overlay_pixmap)
                painter.setOpacity(1.0)

            for line in self._measurements:
                if line.measurement_id is not None and line.measurement_id == self._selected_measurement_id:
                    painter.setPen(QPen(QColor(255, 215, 0), 4))
                else:
                    painter.setPen(QPen(QColor(255, 80, 80), 2))
                p1 = self.image_to_widget(QPointF(line.x1, line.y1))
                p2 = self.image_to_widget(QPointF(line.x2, line.y2))
                painter.drawLine(p1, p2)

            if self._calibration_points:
                painter.setPen(QPen(QColor(255, 215, 0), 2))
                for point in self._calibration_points:
                    widget_point = self.image_to_widget(point)
                    painter.drawEllipse(widget_point, 4, 4)

            if self._calibration_preview:
                painter.setPen(QPen(QColor(255, 215, 0), 2, Qt.DashLine))
                p1 = self.image_to_widget(QPointF(self._calibration_preview[0], self._calibration_preview[1]))
                p2 = self.image_to_widget(QPointF(self._calibration_preview[2], self._calibration_preview[3]))
                painter.drawLine(p1, p2)

            if self._temp_line:
                painter.setPen(QPen(QColor(80, 220, 120), 2, Qt.DashLine))
                p1 = self.image_to_widget(QPointF(self._temp_line[0], self._temp_line[1]))
                p2 = self.image_to_widget(QPointF(self._temp_line[2], self._temp_line[3]))
                painter.drawLine(p1, p2)
            # Draw calibration label if present and we have calibration points
            if self._calibration_label_mm is not None and len(self._calibration_points) >= 2:
                try:
                    p1 = self.image_to_widget(self._calibration_points[0])
                    p2 = self.image_to_widget(self._calibration_points[1])
                    mid_x = (p1.x() + p2.x()) / 2.0
                    mid_y = (p1.y() + p2.y()) / 2.0
                    label = f"{self._calibration_label_mm:.3f} mm"
                    fm = painter.fontMetrics()
                    text_w = fm.horizontalAdvance(label)
                    text_h = fm.height()
                    padding = 6
                    rect_x = mid_x + 6
                    rect_y = mid_y - text_h - 6
                    # clamp rectangle inside widget
                    rect_x = max(4, min(rect_x, self.width() - text_w - padding))
                    rect_y = max(4, min(rect_y, self.height() - text_h - padding))
                    from PySide6.QtCore import QRectF
                    bg_rect = QRectF(rect_x - 4, rect_y - 2, text_w + padding, text_h + 4)
                    painter.setRenderHint(QPainter.TextAntialiasing)
                    painter.setPen(Qt.NoPen)
                    painter.setBrush(QColor(0, 0, 0, 160))
                    painter.drawRoundedRect(bg_rect, 4, 4)
                    painter.setPen(QPen(QColor(255, 255, 255)))
                    painter.drawText(int(rect_x), int(rect_y + text_h - fm.descent()), label)
                except Exception:
                    pass

            # Draw measurement labels if requested
            if self._show_measurement_labels and self._measurement_label_texts:
                try:
                    fm = painter.fontMetrics()
                    padding = 6
                    from PySide6.QtCore import QRectF
                    
                    # Track drawn label rects to avoid overlaps
                    drawn_rects: list[QRectF] = []
                    
                    for line in self._measurements:
                        if line.measurement_id is None:
                            continue
                        text = self._measurement_label_texts.get(line.measurement_id)
                        if not text:
                            continue
                        p1 = self.image_to_widget(QPointF(line.x1, line.y1))
                        p2 = self.image_to_widget(QPointF(line.x2, line.y2))
                        # Skip labels for lines completely outside the widget viewport
                        min_x = min(p1.x(), p2.x())
                        max_x = max(p1.x(), p2.x())
                        min_y = min(p1.y(), p2.y())
                        max_y = max(p1.y(), p2.y())
                        buf = 8
                        w = self.width()
                        h = self.height()
                        if max_x < -buf or min_x > w + buf or max_y < -buf or min_y > h + buf:
                            continue
                        mid_x = (p1.x() + p2.x()) / 2.0
                        mid_y = (p1.y() + p2.y()) / 2.0
                        text_w = fm.horizontalAdvance(text)
                        text_h = fm.height()
                        
                        # Try multiple positions: right, below, above, left
                        candidate_positions = [
                            (mid_x + 6, mid_y - text_h - 6),      # right-above
                            (mid_x + 6, mid_y + 6),               # right-below
                            (mid_x - text_w - 6, mid_y + 6),      # left-below
                            (mid_x - text_w - 6, mid_y - text_h - 6),  # left-above
                        ]
                        
                        rect_x, rect_y = None, None
                        for cand_x, cand_y in candidate_positions:
                            # Clamp to bounds
                            test_x = max(4, min(cand_x, w - text_w - padding))
                            test_y = max(4, min(cand_y, h - text_h - padding))
                            test_rect = QRectF(test_x - 4, test_y - 2, text_w + padding, text_h + 4)
                            
                            # Check for collision with already-drawn labels
                            collision = False
                            for drawn_rect in drawn_rects:
                                if test_rect.intersects(drawn_rect):
                                    collision = True
                                    break
                            
                            if not collision:
                                rect_x, rect_y = test_x, test_y
                                break
                        
                        # If all positions collide, use the first one anyway (but slightly offset)
                        if rect_x is None:
                            rect_x = max(4, min(mid_x + 6, w - text_w - padding))
                            rect_y = max(4, min(mid_y - text_h - 6, h - text_h - padding))
                        
                        bg_rect = QRectF(rect_x - 4, rect_y - 2, text_w + padding, text_h + 4)
                        drawn_rects.append(bg_rect)
                        
                        painter.setRenderHint(QPainter.TextAntialiasing)
                        painter.setPen(Qt.NoPen)
                        painter.setBrush(QColor(0, 0, 0, 160))
                        painter.drawRoundedRect(bg_rect, 4, 4)
                        painter.setPen(QPen(QColor(255, 255, 255)))
                        painter.drawText(int(rect_x), int(rect_y + text_h - fm.descent()), text)
                except Exception:
                    pass
        finally:
            painter.end()

    def sizeHint(self):  # type: ignore[override]
        return self.minimumSizeHint()

    def minimumSizeHint(self):  # type: ignore[override]
        return super().minimumSizeHint().expandedTo(self.size())

    def widget_to_image(self, point: QPointF) -> QPointF:
        if not self._pixmap:
            return QPointF(0.0, 0.0)
        base_x, base_y, _, _ = self._base_geometry()
        x = (point.x() - base_x) / self._view_scale
        y = (point.y() - base_y) / self._view_scale
        return QPointF(x, y)

    def image_to_widget(self, point: QPointF) -> QPointF:
        if not self._pixmap:
            return QPointF(0.0, 0.0)
        base_x, base_y, _, _ = self._base_geometry()
        return QPointF(base_x + point.x() * self._view_scale, base_y + point.y() * self._view_scale)

    def _base_geometry(self) -> tuple[float, float, float, float]:
        if not self._pixmap:
            return 0.0, 0.0, 0.0, 0.0
        width = self._pixmap.width() * self._view_scale
        height = self._pixmap.height() * self._view_scale
        x = (self.width() - width) / 2 + self._pan_offset.x()
        y = (self.height() - height) / 2 + self._pan_offset.y()
        return x, y, width, height

    def _scaled_rect(self, width: float, height: float, x: float, y: float, scale: float):
        return QRect(int(round(x)), int(round(y)), int(round(width * scale)), int(round(height * scale)))

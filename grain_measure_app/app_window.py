from PySide6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QGridLayout, QToolBar,
    QFileDialog, QMessageBox, QTableWidget, QTableWidgetItem, QHeaderView, QInputDialog,
    QSlider, QCheckBox, QSizePolicy, QAbstractItemView, QToolButton, QLabel, QPushButton
)
from PySide6.QtGui import QAction
from PySide6.QtCore import Qt, QSignalBlocker, QTimer
from functools import partial
import math
import os
import cv2
import numpy as np

try:
    from .image_viewer import ImageViewer, MeasurementLine
    from .measurements import MeasurementsManager
    from .measurements import Measurement as StoredMeasurement
    from .calibration import CalibrationData, from_pixel_and_real
    from . import export_csv
except ImportError:
    from image_viewer import ImageViewer, MeasurementLine
    from measurements import MeasurementsManager
    from measurements import Measurement as StoredMeasurement
    from calibration import CalibrationData, from_pixel_and_real
    import export_csv


class AppWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Grain Measure App - Version 1")
        self.calibration: CalibrationData | None = None
        self.measurements = MeasurementsManager()
        self._measurement_points: list[tuple[float, float]] = []
        self._intersects_line_points: list[tuple[float, float]] = []
        self._intersects_points: list[tuple[float, float]] = []
        self._intersects_line_complete = False
        self._selected_measurement_id: int | None = None
        self.overlay_scale: float = 1.0

        # Debounce timers for smooth slider response
        self._sample_enhancement_timer = QTimer()
        self._sample_enhancement_timer.setSingleShot(True)
        self._sample_enhancement_timer.timeout.connect(self._apply_sample_enhancement)
        self._sample_enhancement_pending: tuple[float, float, float] | None = None

        self._ref_enhancement_timer = QTimer()
        self._ref_enhancement_timer.setSingleShot(True)
        self._ref_enhancement_timer.timeout.connect(self._apply_ref_enhancement)
        self._ref_enhancement_pending: tuple[float, float, float] | None = None

        self._init_ui()

    def _init_ui(self) -> None:
        # Top toolbar: only Export CSV
        toolbar = QToolBar("Main")
        self.addToolBar(toolbar)
        export_action = QAction("💾 Export CSV", self)
        export_action.triggered.connect(self.export_csv)
        toolbar.addAction(export_action)

        central = QWidget()
        self.setCentralWidget(central)

        main_layout = QVBoxLayout(central)

        # Top: viewers with separate control columns
        viewers_and_controls_layout = QHBoxLayout()

        # Left column: reference viewer with reference-specific buttons (using grid for better row control)
        left_column = QGridLayout()
        
        # Row 0: Reference image buttons (above viewer)
        ref_buttons_layout = QHBoxLayout()
        load_ref_action = QAction("Load reference", self)
        load_ref_action.triggered.connect(self.load_reference_image)
        load_ref_btn = QPushButton("📁 Load reference")
        load_ref_btn.clicked.connect(self.load_reference_image)
        ref_buttons_layout.addWidget(load_ref_btn)

        calibrate_action = QAction("Calibrate", self)
        calibrate_action.triggered.connect(self.start_calibration)
        calibrate_btn = QPushButton("📏 Calibrate")
        calibrate_btn.clicked.connect(self.start_calibration)
        ref_buttons_layout.addWidget(calibrate_btn)

        scale_sample_action = QAction("Scale sample", self)
        scale_sample_action.setCheckable(True)
        scale_sample_action.triggered.connect(self._on_scale_sample_toggled)
        self.scale_sample_action = scale_sample_action
        scale_btn = QPushButton("↕️ Scale sample")
        scale_btn.setCheckable(True)
        scale_btn.toggled.connect(self._on_scale_sample_toggled)
        self.scale_sample_button = scale_btn
        ref_buttons_layout.addWidget(scale_btn)

        move_overlay_action = QAction("Move overlay", self)
        move_overlay_action.setCheckable(True)
        move_overlay_action.triggered.connect(self._on_move_overlay_toggled)
        self.move_overlay_action = move_overlay_action
        move_overlay_btn = QPushButton("🔄 Move overlay")
        move_overlay_btn.setCheckable(True)
        move_overlay_btn.toggled.connect(self._on_move_overlay_toggled)
        self.move_overlay_button = move_overlay_btn
        ref_buttons_layout.addWidget(move_overlay_btn)
        
        # Reference filename label (right-aligned)
        self.ref_filename_label = QLabel("(no file)")
        self.ref_filename_label.setStyleSheet("color: #888; font-size: 11px;")
        self.ref_filename_label.setMaximumWidth(200)
        self.ref_filename_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        ref_buttons_layout.addStretch()
        ref_buttons_layout.addWidget(self.ref_filename_label)
        
        left_column.addLayout(ref_buttons_layout, 0, 0)

        # Row 1: Image viewer (stretched)
        self.ref_viewer = ImageViewer()
        left_column.addWidget(self.ref_viewer, 1, 0)
        left_column.setRowStretch(1, 1)  # Row 1 gets all available vertical space

        # Row 2: Reference enhancement controls (under reference viewer)
        ref_enh_layout = QHBoxLayout()
        ref_enh_layout.addWidget(QLabel("Ref Brightness:"))
        self.ref_brightness_slider = QSlider(Qt.Horizontal)
        self.ref_brightness_slider.setRange(-100, 100)
        self.ref_brightness_slider.setValue(0)
        self.ref_brightness_slider.setMaximumWidth(100)
        self.ref_brightness_slider.valueChanged.connect(self._on_ref_enhancement_changed)
        ref_enh_layout.addWidget(self.ref_brightness_slider)
        self.ref_brightness_value_label = QLabel("0")
        self.ref_brightness_value_label.setMaximumWidth(25)
        ref_enh_layout.addWidget(self.ref_brightness_value_label)
        ref_enh_layout.addWidget(QLabel("Ref Contrast:"))
        self.ref_contrast_slider = QSlider(Qt.Horizontal)
        self.ref_contrast_slider.setRange(50, 300)
        self.ref_contrast_slider.setValue(100)
        self.ref_contrast_slider.setMaximumWidth(100)
        self.ref_contrast_slider.valueChanged.connect(self._on_ref_enhancement_changed)
        ref_enh_layout.addWidget(self.ref_contrast_slider)
        self.ref_contrast_value_label = QLabel("1.0")
        self.ref_contrast_value_label.setMaximumWidth(25)
        ref_enh_layout.addWidget(self.ref_contrast_value_label)
        ref_enh_layout.addWidget(QLabel("Ref Sat:"))
        self.ref_saturation_slider = QSlider(Qt.Horizontal)
        self.ref_saturation_slider.setRange(0, 200)
        self.ref_saturation_slider.setValue(100)
        self.ref_saturation_slider.setMaximumWidth(100)
        self.ref_saturation_slider.valueChanged.connect(self._on_ref_enhancement_changed)
        ref_enh_layout.addWidget(self.ref_saturation_slider)
        self.ref_saturation_value_label = QLabel("1.0")
        self.ref_saturation_value_label.setMaximumWidth(25)
        ref_enh_layout.addWidget(self.ref_saturation_value_label)
        ref_reset_btn = QPushButton("Reset")
        ref_reset_btn.setMaximumWidth(50)
        ref_reset_btn.clicked.connect(self._on_reset_ref_enhancements)
        ref_enh_layout.addWidget(ref_reset_btn)
        left_column.addLayout(ref_enh_layout, 2, 0)
        viewers_and_controls_layout.addLayout(left_column)

        # Right column: sample viewer + sample-specific buttons + enhancement controls (using grid for better row control)
        right_column = QGridLayout()
        
        # Row 0: Sample image buttons (above viewer)
        sample_buttons_layout = QHBoxLayout()
        load_sample_btn = QPushButton("📁 Load sample")
        load_sample_btn.clicked.connect(self.load_sample_image)
        sample_buttons_layout.addWidget(load_sample_btn)

        measure_action = QAction("Measure", self)
        measure_action.setCheckable(True)
        measure_action.triggered.connect(self._on_measure_toggled)
        self.measure_action = measure_action
        measure_btn = QPushButton("📐 Measure")
        measure_btn.setCheckable(True)
        measure_btn.toggled.connect(self._on_measure_toggled)
        self.measure_button = measure_btn
        sample_buttons_layout.addWidget(measure_btn)

        intersects_action = QAction("Intersects", self)
        intersects_action.setCheckable(True)
        intersects_action.triggered.connect(self._on_intersects_toggled)
        self.intersects_action = intersects_action
        intersects_btn = QPushButton("📍 Intersects")
        intersects_btn.setCheckable(True)
        intersects_btn.toggled.connect(self._on_intersects_toggled)
        self.intersects_button = intersects_btn
        sample_buttons_layout.addWidget(intersects_btn)
        # A contextual "Done" checkmark shown only during an active intersects session
        done_btn = QPushButton("✔ Done")
        done_btn.setVisible(False)
        done_btn.setMaximumWidth(80)
        done_btn.clicked.connect(self._on_intersects_done_clicked)
        self.intersects_done_button = done_btn
        sample_buttons_layout.addWidget(done_btn)
        sample_buttons_layout.addStretch()
        
        # Sample filename label (right-aligned)
        self.sample_filename_label = QLabel("(no file)")
        self.sample_filename_label.setStyleSheet("color: #888; font-size: 11px;")
        self.sample_filename_label.setMaximumWidth(200)
        self.sample_filename_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        sample_buttons_layout.addWidget(self.sample_filename_label)
        
        right_column.addLayout(sample_buttons_layout, 0, 0)
        
        # Row 1: Image viewer (stretched)
        self.sample_viewer = ImageViewer()
        right_column.addWidget(self.sample_viewer, 1, 0)
        right_column.setRowStretch(1, 1)  # Row 1 gets all available vertical space

        # Row 2: enhancement controls (brightness, contrast, saturation) - on the right under sample
        enhancement_layout = QHBoxLayout()
        
        # Brightness
        enhancement_layout.addWidget(QLabel("Brightness:"))
        self.brightness_slider = QSlider(Qt.Horizontal)
        self.brightness_slider.setRange(-100, 100)
        self.brightness_slider.setValue(0)
        self.brightness_slider.setMaximumWidth(100)
        self.brightness_slider.valueChanged.connect(self._on_enhancement_changed)
        enhancement_layout.addWidget(self.brightness_slider)
        self.brightness_value_label = QLabel("0")
        self.brightness_value_label.setMaximumWidth(25)
        enhancement_layout.addWidget(self.brightness_value_label)
        
        # Contrast
        enhancement_layout.addWidget(QLabel("Contrast:"))
        self.contrast_slider = QSlider(Qt.Horizontal)
        self.contrast_slider.setRange(50, 300)
        self.contrast_slider.setValue(100)
        self.contrast_slider.setMaximumWidth(100)
        self.contrast_slider.valueChanged.connect(self._on_enhancement_changed)
        enhancement_layout.addWidget(self.contrast_slider)
        self.contrast_value_label = QLabel("1.0")
        self.contrast_value_label.setMaximumWidth(25)
        enhancement_layout.addWidget(self.contrast_value_label)
        
        # Saturation
        enhancement_layout.addWidget(QLabel("Saturation:"))
        self.saturation_slider = QSlider(Qt.Horizontal)
        self.saturation_slider.setRange(0, 200)
        self.saturation_slider.setValue(100)
        self.saturation_slider.setMaximumWidth(100)
        self.saturation_slider.valueChanged.connect(self._on_enhancement_changed)
        enhancement_layout.addWidget(self.saturation_slider)
        self.saturation_value_label = QLabel("1.0")
        self.saturation_value_label.setMaximumWidth(25)
        enhancement_layout.addWidget(self.saturation_value_label)
        
        # Reset button
        reset_btn = QPushButton("Reset")
        reset_btn.setMaximumWidth(50)
        reset_btn.clicked.connect(self._on_reset_enhancements)
        enhancement_layout.addWidget(reset_btn)

        right_column.addLayout(enhancement_layout, 2, 0)

        # Row 3: statistics and measurement labels (under enhancement)
        stats_layout = QHBoxLayout()
        self.stats_label = QLabel("Measurements: 0 | Line avg: — | Intersects avg: —")
        stats_layout.addWidget(self.stats_label)
        stats_layout.addStretch()
        # Toggle to show measurement labels on sample image
        self.show_labels_checkbox = QCheckBox("Show measurement labels")
        self.show_labels_checkbox.setChecked(False)
        self.show_labels_checkbox.stateChanged.connect(lambda s: self._on_show_measurement_labels(bool(s)))
        stats_layout.addWidget(self.show_labels_checkbox)
        # Toggle to include sample number in labels
        self.show_sample_numbers_checkbox = QCheckBox("Include sample number")
        self.show_sample_numbers_checkbox.setChecked(False)
        self.show_sample_numbers_checkbox.stateChanged.connect(lambda s: self._update_measurement_labels())
        stats_layout.addWidget(self.show_sample_numbers_checkbox)
        # Export sample image button
        export_sample_btn = QPushButton("💾 Export sample")
        export_sample_btn.setMaximumWidth(120)
        export_sample_btn.clicked.connect(self.export_sample_image)
        stats_layout.addWidget(export_sample_btn)
        stats_layout.addStretch()
        right_column.addLayout(stats_layout, 3, 0)

        # Row 4: Measurement table with separate line/intersects result columns
        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(["ID", "Type", "Line length µm", "Intersects", "Avg size µm", "Delete"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.itemSelectionChanged.connect(self._on_table_selection_changed)
        self.table.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
        self.table.setMaximumHeight(220)
        # Slightly reduce row height for a more compact table
        try:
            self.table.verticalHeader().setDefaultSectionSize(22)
        except Exception:
            pass
        right_column.addWidget(self.table, 4, 0)
        
        viewers_and_controls_layout.addLayout(right_column)

        main_layout.addLayout(viewers_and_controls_layout, 8)

        # Row 3: overlay controls (place under reference viewer on the left)
        overlay_layout = QHBoxLayout()
        self.overlay_checkbox = QCheckBox("Overlay sample on reference")
        self.overlay_checkbox.stateChanged.connect(self._on_overlay_toggled)
        overlay_layout.addWidget(self.overlay_checkbox)

        self.opacity_slider = QSlider(Qt.Horizontal)
        self.opacity_slider.setRange(0, 100)
        self.opacity_slider.setValue(50)
        self.opacity_slider.setEnabled(False)
        self.opacity_slider.valueChanged.connect(self._on_opacity_changed)
        # keep opacity slider short and only in left column
        self.opacity_slider.setMaximumWidth(150)
        overlay_layout.addWidget(self.opacity_slider)
        left_column.addLayout(overlay_layout, 3, 0)

        # connect overlay scale changes (update table/display when user scales)
        try:
            self.ref_viewer.connect_overlay_scale_changed(self._on_overlay_scale_changed)
        except Exception:
            pass

    def load_reference_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open reference image")
        if not path:
            return
        self.ref_viewer.load_image(path)
        # Update reference filename label
        filename = os.path.basename(path)
        self.ref_filename_label.setText(filename)
        self.ref_filename_label.setToolTip(path)  # Full path on hover
        # if overlay mode active and sample loaded, set overlay
        if self.overlay_checkbox.isChecked() and self.sample_viewer._image_path:
            self.ref_viewer.set_overlay_image(self.sample_viewer._image_path)
            self.ref_viewer.set_move_overlay_enabled(self.move_overlay_action.isChecked())

    def load_sample_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open sample image")
        if not path:
            return
        self.sample_viewer.load_image(path)
        # Update sample filename label
        filename = os.path.basename(path)
        self.sample_filename_label.setText(filename)
        self.sample_filename_label.setToolTip(path)  # Full path on hover
        self._refresh_measurement_overlay()
        # if overlay mode active, set as overlay on reference
        if self.overlay_checkbox.isChecked() and self.ref_viewer._image_path:
            self.ref_viewer.set_overlay_image(path)
            self.opacity_slider.setEnabled(True)

    def calibrate(self) -> None:
        QMessageBox.information(self, "Calibrate", "Use 'Calibrate' from the toolbar to start.")

    def start_calibration(self) -> None:
        if not self.ref_viewer._pixmap:
            QMessageBox.warning(self, "Calibrate", "Load a reference image first.")
            return

        self._calib_points: list[tuple[float, float]] = []
        self.ref_viewer.clear_calibration_marks()
        # clear any previous calibration label shown on reference
        try:
            self.ref_viewer.set_calibration_label_mm(None)
        except Exception:
            pass
        try:
            self.ref_viewer.connect_point(self._on_ref_point_clicked)
        except Exception:
            self.ref_viewer.on_point_clicked = self._on_ref_point_clicked

        try:
            self.ref_viewer.connect_hover(self._on_ref_hover)
        except Exception:
            self.ref_viewer.on_hover_point = self._on_ref_hover

        self.sample_viewer.enable_point_selection(False)
        self.sample_viewer.disconnect_point()
        self.ref_viewer.set_view_mode("pick")
        self.ref_viewer.enable_point_selection(True)
        self.statusBar().showMessage("Calibration: click two points on the reference image")

    def _on_ref_hover(self, x: float, y: float) -> None:
        if len(self._calib_points) == 1:
            self.ref_viewer.set_calibration_preview(self._calib_points[0], (x, y))

    def _on_ref_point_clicked(self, x: float, y: float) -> None:
        self._calib_points.append((x, y))
        self.ref_viewer.add_calibration_point((x, y))
        if len(self._calib_points) == 1:
            self.ref_viewer.set_calibration_preview(self._calib_points[0], self._calib_points[0])
            self.statusBar().showMessage("Calibration: first point recorded, click second point")
            return

        # second point
        x1, y1 = self._calib_points[0]
        x2, y2 = self._calib_points[1]
        self.ref_viewer.set_calibration_preview((x1, y1), (x2, y2))
        dx = x2 - x1
        dy = y2 - y1
        pixel_distance = math.hypot(dx, dy)

        if pixel_distance <= 0.0:
            QMessageBox.warning(self, "Calibration", "Points are identical. Try again.")
            self._finish_calibration()
            return

        # getDouble signature: (parent, title, label, value=0.0, min=-2147483647, max=2147483647, decimals=1)
        value, ok = QInputDialog.getDouble(self, "Known distance", "Enter known distance between points (mm):", 0.0, 0.0, 1e9, 6)
        if not ok or value <= 0.0:
            QMessageBox.warning(self, "Calibration", "Invalid distance entered. Calibration cancelled.")
            self._finish_calibration()
            return

        try:
            self.calibration = from_pixel_and_real(pixel_distance, float(value))
            self.statusBar().showMessage(f"Calibration set: {self.calibration.pixels_per_mm:.2f} px/mm, {self.calibration.um_per_pixel:.3f} µm/px")
            QMessageBox.information(self, "Calibration", "Calibration successful.")
        except Exception as e:
            QMessageBox.critical(self, "Calibration", f"Failed to compute calibration:\n{e}")

        # show calibration label on reference viewer
        try:
            self.ref_viewer.set_calibration_label_mm(self.calibration.known_distance_mm)
        except Exception:
            pass

        self._finish_calibration()

    def _finish_calibration(self) -> None:
        try:
            self.ref_viewer.disconnect_point()
        except Exception:
            if hasattr(self.ref_viewer, "on_point_clicked"):
                delattr(self.ref_viewer, "on_point_clicked")
        try:
            self.ref_viewer.disconnect_hover()
        except Exception:
            if hasattr(self.ref_viewer, "on_hover_point"):
                delattr(self.ref_viewer, "on_hover_point")
        self.ref_viewer.enable_point_selection(False)
        self.ref_viewer.set_view_mode("navigate")
        # ensure sample viewer visibility when finishing calibration
        self._restore_default_interaction_modes()

    def measure(self) -> None:
        QMessageBox.information(self, "Measure", "Use the Measure toggle in the toolbar to start line measurement.")

    def _on_measure_toggled(self, checked: bool) -> None:
        if checked:
            if self.intersects_button.isChecked():
                self._cancel_intersects_session()
                self._set_button_checked(self.intersects_button, False)
            if not self.sample_viewer._pixmap:
                QMessageBox.warning(self, "Measure", "Load a sample image first.")
                self.measure_action.setChecked(False)
                return
            if not self.calibration:
                QMessageBox.warning(self, "Measure", "Calibrate first before measuring.")
                self.measure_action.setChecked(False)
                return

            self._measurement_points = []
            self.sample_viewer.enable_point_selection(True)
            self.sample_viewer.connect_point(self._on_sample_point_clicked)
            self.sample_viewer.disconnect_hover()
            self.sample_viewer.set_view_mode("pick")
            self.statusBar().showMessage("Measurement: click two points on the sample image")
        else:
            self._measurement_points = []
            self.sample_viewer.clear_measurement_preview()
            self.sample_viewer.disconnect_point()
            self.sample_viewer.enable_point_selection(False)
            self.sample_viewer.set_view_mode("navigate")
            self.statusBar().showMessage("Measurement mode off")

    def _on_intersects_toggled(self, checked: bool) -> None:
        if checked:
            if self.measure_button.isChecked():
                self._measurement_points = []
                self.sample_viewer.clear_measurement_preview()
                self.sample_viewer.disconnect_point()
                self._set_button_checked(self.measure_button, False)
            if not self.sample_viewer._pixmap:
                QMessageBox.warning(self, "Intersects", "Load a sample image first.")
                self._set_button_checked(self.intersects_button, False)
                return
            if not self.calibration:
                QMessageBox.warning(self, "Intersects", "Calibrate first before measuring.")
                self._set_button_checked(self.intersects_button, False)
                return

            self._intersects_line_points = []
            self._intersects_points = []
            self._intersects_line_complete = False
            self.sample_viewer.enable_point_selection(True)
            self.sample_viewer.connect_point(self._on_intersects_point_clicked)
            self.sample_viewer.connect_hover(self._on_intersects_hover)
            self.sample_viewer.set_view_mode("pick")
            self.sample_viewer.clear_measurement_preview()
            self.sample_viewer.set_intersects_preview(None, None, None)
            try:
                self.intersects_done_button.setVisible(True)
            except Exception:
                pass
            self.statusBar().showMessage("Intersects: click two points to draw the line, then click all grain intersects")
        else:
            self._finish_intersects_session(save=True)

    def _on_sample_point_clicked(self, x: float, y: float) -> None:
        self._measurement_points.append((x, y))
        if len(self._measurement_points) == 1:
            self.sample_viewer.set_measurement_preview(self._measurement_points[0], self._measurement_points[0])
            self.statusBar().showMessage("Measurement: first point recorded, click second point")
            return

        x1, y1 = self._measurement_points[0]
        x2, y2 = self._measurement_points[1]
        pixel_length = math.hypot(x2 - x1, y2 - y1)
        if pixel_length <= 0.0:
            QMessageBox.warning(self, "Measure", "Points are identical. Try again.")
            self._measurement_points = []
            self.sample_viewer.clear_measurement_preview()
            return

        if not self.calibration:
            QMessageBox.warning(self, "Measure", "Calibration is missing.")
            return

        # apply current overlay/sample scale so measurements reflect scaled overlay
        scaled_pixels = pixel_length * float(self.overlay_scale)
        length_mm = scaled_pixels * self.calibration.mm_per_pixel
        length_um = scaled_pixels * self.calibration.um_per_pixel

        measurement = StoredMeasurement(
            id=0,
            measurement_type="line",
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
            pixel_length=pixel_length,
            length_mm=length_mm,
            length_um=length_um,
        )
        self.measurements.add(measurement)
        self._measurement_points = []
        self.sample_viewer.clear_measurement_preview()
        self._refresh_measurement_overlay()
        self.refresh_table()
        self.statusBar().showMessage(f"Added measurement {measurement.id}: {length_um:.2f} µm")

    def _cancel_intersects_session(self) -> None:
        self._intersects_line_points = []
        self._intersects_points = []
        self._intersects_line_complete = False
        self.sample_viewer.clear_measurement_preview()
        self.sample_viewer.clear_intersects_preview()
        self.sample_viewer.disconnect_point()
        self.sample_viewer.disconnect_hover()
        self.sample_viewer.enable_point_selection(False)
        self.sample_viewer.set_view_mode("navigate")
        try:
            self.intersects_done_button.setVisible(False)
        except Exception:
            pass

    def _finish_intersects_session(self, save: bool) -> None:
        if save and self._intersects_line_complete and self._intersects_points:
            x1, y1 = self._intersects_line_points[0]
            x2, y2 = self._intersects_line_points[1]
            pixel_length = math.hypot(x2 - x1, y2 - y1)
            if pixel_length <= 0.0:
                QMessageBox.warning(self, "Intersects", "The line is too short to measure.")
            elif not self.calibration:
                QMessageBox.warning(self, "Intersects", "Calibration is missing.")
            else:
                count = len(self._intersects_points)
                scaled_pixels = pixel_length * float(self.overlay_scale)
                length_mm = scaled_pixels * self.calibration.mm_per_pixel
                length_um = scaled_pixels * self.calibration.um_per_pixel
                grain_size_mm = length_mm / count
                grain_size_um = length_um / count

                measurement = StoredMeasurement(
                    id=0,
                    measurement_type="intersects",
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    pixel_length=pixel_length,
                    length_mm=length_mm,
                    length_um=length_um,
                    intersect_points=tuple(self._intersects_points),
                    grain_size_mm=grain_size_mm,
                    grain_size_um=grain_size_um,
                )
                self.measurements.add(measurement)
                self._refresh_measurement_overlay()
                self.refresh_table()
                self.statusBar().showMessage(
                    f"Added intersects measurement {measurement.id}: {count} points, {grain_size_um:.2f} µm"
                )
        elif save and self._intersects_line_complete and not self._intersects_points:
            QMessageBox.warning(self, "Intersects", "Mark at least one intersect point before finishing.")

        self._cancel_intersects_session()
        try:
            self.intersects_done_button.setVisible(False)
        except Exception:
            pass
        # ensure intersects toggle is cleared
        try:
            self.intersects_button.setChecked(False)
        except Exception:
            try:
                self._set_button_checked(self.intersects_button, False)
            except Exception:
                pass

    def _on_intersects_hover(self, x: float, y: float) -> None:
        if len(self._intersects_line_points) == 1 and not self._intersects_line_complete:
            self.sample_viewer.set_intersects_preview(self._intersects_line_points[0], (x, y), [])

    def _distance_to_segment(self, px: float, py: float, x1: float, y1: float, x2: float, y2: float) -> float:
        dx = x2 - x1
        dy = y2 - y1
        if dx == 0.0 and dy == 0.0:
            return math.hypot(px - x1, py - y1)
        t = ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)
        t = max(0.0, min(1.0, t))
        closest_x = x1 + t * dx
        closest_y = y1 + t * dy
        return math.hypot(px - closest_x, py - closest_y)

    def _on_intersects_point_clicked(self, x: float, y: float) -> None:
        if len(self._intersects_line_points) < 2:
            self._intersects_line_points.append((x, y))
            if len(self._intersects_line_points) == 1:
                self.sample_viewer.set_intersects_preview(self._intersects_line_points[0], self._intersects_line_points[0], [])
                self.statusBar().showMessage("Intersects: first line point recorded, click second point")
            else:
                self._intersects_line_complete = True
                self.sample_viewer.set_intersects_preview(self._intersects_line_points[0], self._intersects_line_points[1], [])
                self.statusBar().showMessage("Intersects: line drawn, click grain intersection points on the line")
            return

        x1, y1 = self._intersects_line_points[0]
        x2, y2 = self._intersects_line_points[1]
        if self._distance_to_segment(x, y, x1, y1, x2, y2) > 12.0:
            QMessageBox.information(self, "Intersects", "Click on the drawn line to mark an intersect point.")
            return

        self._intersects_points.append((x, y))
        self.sample_viewer.set_intersects_preview(self._intersects_line_points[0], self._intersects_line_points[1], self._intersects_points)
        count = len(self._intersects_points)
        result_um = self._measurement_result_um_for_intersects_preview()
        self.statusBar().showMessage(f"Intersects: {count} point(s) marked | Grain size: {result_um:.2f} µm")

    def _on_intersects_done_clicked(self) -> None:
        """Handler for the contextual Done button next to Intersects.
        Finishes the current intersects session (saving if possible) and hides the Done button.
        """
        # delegate to existing finish logic
        self._finish_intersects_session(save=True)
        try:
            self.intersects_done_button.setVisible(False)
        except Exception:
            pass

    def _measurement_result_um_for_intersects_preview(self) -> float:
        if len(self._intersects_line_points) < 2 or not self.calibration or not self._intersects_points:
            return 0.0
        x1, y1 = self._intersects_line_points[0]
        x2, y2 = self._intersects_line_points[1]
        pixel_length = math.hypot(x2 - x1, y2 - y1)
        scaled_pixels = pixel_length * float(self.overlay_scale)
        length_um = scaled_pixels * self.calibration.um_per_pixel
        return length_um / len(self._intersects_points)

    def export_csv(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Export measurements", filter="CSV Files (*.csv)")
        if not path:
            return
        try:
            export_csv.export_measurements_csv(path, self.measurements.get_all(), self.calibration, overlay_scale=self.overlay_scale)
            QMessageBox.information(self, "Export", f"Exported CSV to:\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to export CSV:\n{e}")

    def export_sample_image(self) -> None:
        """Export sample image with annotations (measurement lines and labels) if enabled."""
        if not self.sample_viewer._image_path:
            QMessageBox.warning(self, "Warning", "No sample image loaded")
            return
        
        path, _ = QFileDialog.getSaveFileName(self, "Export sample image", filter="PNG Files (*.png);;JPEG Files (*.jpg);;BMP Files (*.bmp)")
        if not path:
            return
        
        try:
            # Get the original image
            image_bgr = self.sample_viewer._image_cache_bgr
            if image_bgr is None:
                QMessageBox.warning(self, "Warning", "Unable to load image data")
                return
            
            # Create a copy to draw on
            export_image = image_bgr.copy()
            
            # Draw measurement lines if any exist
            for line in self.sample_viewer._measurements:
                pt1 = (int(line.x1), int(line.y1))
                pt2 = (int(line.x2), int(line.y2))
                # Draw line in red (BGR format)
                cv2.line(export_image, pt1, pt2, (80, 80, 255), 2)
                if line.measurement_type == "intersects" and line.intersect_points:
                    for idx, (px, py) in enumerate(line.intersect_points, start=1):
                        center = (int(px), int(py))
                        cv2.circle(export_image, center, 4, (80, 220, 120), -1)
                        cv2.putText(export_image, str(idx), (center[0] + 6, center[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Draw measurement labels if enabled
            if self.sample_viewer._show_measurement_labels and self.sample_viewer._measurement_label_texts:
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.6
                font_color = (255, 255, 255)  # White in BGR
                bg_color = (0, 0, 0)  # Black background
                thickness = 1
                
                for line in self.sample_viewer._measurements:
                    if line.measurement_id is None:
                        continue
                    text = self.sample_viewer._measurement_label_texts.get(line.measurement_id)
                    if not text:
                        continue
                    
                    # Calculate midpoint
                    mid_x = int((line.x1 + line.x2) / 2)
                    mid_y = int((line.y1 + line.y2) / 2)
                    
                    # Get text size
                    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
                    
                    # Draw background rectangle
                    padding = 4
                    cv2.rectangle(export_image, 
                                (mid_x - text_w // 2 - padding, mid_y - text_h - baseline - padding),
                                (mid_x + text_w // 2 + padding, mid_y + baseline + padding),
                                bg_color, -1)
                    
                    # Draw text
                    cv2.putText(export_image, text, (mid_x - text_w // 2, mid_y), 
                              font, font_scale, font_color, thickness)
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            # Try cv2.imwrite first, fallback to manual encoding if it fails
            success = cv2.imwrite(path, export_image)
            if not success:
                # Fallback: use imencode + manual write with proper encoding
                ext = os.path.splitext(path)[1].lower()
                if not ext:
                    ext = '.png'
                ret, buffer = cv2.imencode(ext, export_image)
                if ret:
                    with open(path, 'wb') as f:
                        f.write(buffer)
                    success = True
            
            if success:
                QMessageBox.information(self, "Export", f"Exported sample image to:\n{path}")
            else:
                QMessageBox.warning(self, "Warning", f"Failed to write image to:\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to export sample image:\n{e}")

    def refresh_table(self) -> None:
        current_selection = self._selected_measurement_id
        blocker = QSignalBlocker(self.table)
        rows = self.measurements.count()
        self.table.setRowCount(rows)
        for i, m in enumerate(self.measurements.get_all()):
            self.table.setItem(i, 0, QTableWidgetItem(str(m.id)))
            self.table.setItem(i, 1, QTableWidgetItem(m.measurement_type))
            self.table.setItem(i, 2, QTableWidgetItem(self._measurement_line_length_text(m)))
            self.table.setItem(i, 3, QTableWidgetItem(self._measurement_intersects_text(m)))
            self.table.setItem(i, 4, QTableWidgetItem(self._measurement_avg_size_text(m)))
            delete_button = QToolButton()
            delete_button.setText("×")
            delete_button.setToolTip("Delete measurement")
            delete_button.setAutoRaise(True)
            delete_button.setCursor(Qt.PointingHandCursor)
            delete_button.setStyleSheet(
                "QToolButton { color: #b3261e; font-size: 18px; font-weight: bold; border: none; }"
                "QToolButton:hover { color: #d93025; }"
            )
            delete_button.clicked.connect(partial(self.delete_measurement_by_id, m.id))
            self.table.setCellWidget(i, 5, delete_button)

        self._selected_measurement_id = current_selection if any(m.id == current_selection for m in self.measurements.get_all()) else None
        del blocker

        if self._selected_measurement_id is None:
            self.table.clearSelection()
        else:
            for row_index in range(self.table.rowCount()):
                item = self.table.item(row_index, 0)
                if item and int(item.text()) == self._selected_measurement_id:
                    self.table.selectRow(row_index)
                    break
        self._sync_selected_measurement_highlight()
        self._update_statistics()

        # Also update measurement labels mapping for sample viewer
        self._update_measurement_labels()

    def _update_statistics(self) -> None:
        """Update the statistics panel with average and standard deviation."""
        measurements = self.measurements.get_all()
        count = len(measurements)
        
        if count == 0:
            self.stats_label.setText("Measurements: 0 | Line avg: — | Intersects avg: —")
            return

        line_measurements = [m for m in measurements if m.measurement_type == "line"]
        intersects_measurements = [m for m in measurements if m.measurement_type == "intersects"]

        line_avg = self._mean([m.length_um for m in line_measurements])
        intersects_avg = self._mean([self._measurement_avg_size_um(m) for m in intersects_measurements])

        line_text = f"{line_avg:.2f} µm" if line_avg is not None else "—"
        intersects_text = f"{intersects_avg:.2f} µm" if intersects_avg is not None else "—"

        self.stats_label.setText(
            f"Measurements: {count} | Line avg: {line_text} | Intersects avg: {intersects_text}"
        )

    def delete_measurement_by_id(self, measurement_id: int) -> None:
        if self.measurements.remove_by_id(measurement_id):
            if self._selected_measurement_id == measurement_id:
                self._selected_measurement_id = None
            self.refresh_table()
            self._refresh_measurement_overlay()
        else:
            QMessageBox.warning(self, "Delete measurement", "Could not find the selected measurement.")

    def _on_table_selection_changed(self) -> None:
        selected_rows = self.table.selectionModel().selectedRows()
        if not selected_rows:
            self._selected_measurement_id = None
        else:
            row = selected_rows[0].row()
            item = self.table.item(row, 0)
            self._selected_measurement_id = int(item.text()) if item else None
        self._sync_selected_measurement_highlight()

    def _sync_selected_measurement_highlight(self) -> None:
        self.sample_viewer.set_selected_measurement_id(self._selected_measurement_id)

    def _on_overlay_toggled(self, state: int) -> None:
        enabled = bool(state)
        if enabled:
            # enable overlay; if sample loaded, apply
            if self.sample_viewer._image_path and self.ref_viewer._image_path:
                self.ref_viewer.set_overlay_image(self.sample_viewer._image_path)
                self.opacity_slider.setEnabled(True)
                self.ref_viewer.set_move_overlay_enabled(self.move_overlay_action.isChecked())
                # ensure overlay scale applied
                self.ref_viewer.set_overlay_scale(self.overlay_scale)
            else:
                # no images yet; just enable slider when sample appears
                self.opacity_slider.setEnabled(False)
        else:
            # disable overlay
            self.ref_viewer.clear_overlay()
            self.opacity_slider.setEnabled(False)
            self.ref_viewer.set_move_overlay_enabled(False)

    def _on_opacity_changed(self, value: int) -> None:
        op = float(value) / 100.0
        self.ref_viewer.set_overlay_opacity(op)

    def _on_move_overlay_toggled(self, checked: bool) -> None:
        self.ref_viewer.set_move_overlay_enabled(checked)
        if checked and not self.overlay_checkbox.isChecked():
            self.overlay_checkbox.setChecked(True)

    def _on_scale_sample_toggled(self, checked: bool) -> None:
        # when toggled, the ref_viewer will use mouse wheel to change overlay scale
        self.ref_viewer.set_overlay_scale_mode(checked)
        if checked and not self.overlay_checkbox.isChecked():
            self.overlay_checkbox.setChecked(True)
            self.opacity_slider.setEnabled(True)
        self.statusBar().showMessage("Scale sample: use mouse wheel over reference to adjust (hold Shift for fine scale)")

    def _on_overlay_scale_changed(self, scale: float) -> None:
        try:
            self.overlay_scale = float(scale)
        except Exception:
            self.overlay_scale = 1.0
        # refresh display values that depend on overlay scale
        self.refresh_table()
        self._refresh_measurement_overlay()

    def _refresh_measurement_overlay(self) -> None:
        lines = [
            MeasurementLine(
                m.x1,
                m.y1,
                m.x2,
                m.y2,
                m.id,
                m.measurement_type,
                tuple(getattr(m, "intersect_points", ()) or ()),
            )
            for m in self.measurements.get_all()
        ]
        self.sample_viewer.set_measurements(lines)
        self.sample_viewer.set_selected_measurement_id(self._selected_measurement_id)
        # update labels mapping when overlay refreshed
        self._update_measurement_labels()

    def _on_show_measurement_labels(self, enabled: bool) -> None:
        self.sample_viewer.set_show_measurement_labels(enabled)
        # ensure labels mapping is refreshed
        self._update_measurement_labels()

    def _update_measurement_labels(self) -> None:
        """Compute and send label text mapping for measurements to the sample viewer."""
        mapping: dict[int, str] = {}
        if not self.calibration:
            # nothing to show
            self.sample_viewer.set_measurement_label_texts(mapping)
            return
        for m in self.measurements.get_all():
            result_um = self._measurement_result_um(m)
            if m.measurement_type == "intersects":
                count = self._measurement_points_count(m)
                label = f"{result_um:.2f} µm ({count} pts)"
            else:
                label = f"{result_um:.2f} µm"

            if self.show_sample_numbers_checkbox.isChecked():
                mapping[m.id] = f"{m.id}: {label}"
            else:
                mapping[m.id] = label
        self.sample_viewer.set_measurement_label_texts(mapping)

    def _measurement_points_count(self, measurement: StoredMeasurement) -> int:
        points = getattr(measurement, "intersect_points", ()) or ()
        return len(points) if measurement.measurement_type == "intersects" else 0

    def _mean(self, values: list[float | None]) -> float | None:
        filtered = [value for value in values if value is not None]
        if not filtered:
            return None
        return sum(filtered) / len(filtered)

    def _measurement_line_length_text(self, measurement: StoredMeasurement) -> str:
        return f"{float(measurement.length_um):.2f}" if measurement.length_um is not None else "—"

    def _measurement_intersects_text(self, measurement: StoredMeasurement) -> str:
        if measurement.measurement_type != "intersects":
            return "—"
        return str(self._measurement_points_count(measurement))

    def _measurement_avg_size_um(self, measurement: StoredMeasurement) -> float | None:
        if measurement.measurement_type != "intersects":
            return None
        grain_size_um = getattr(measurement, "grain_size_um", None)
        if grain_size_um is not None:
            return float(grain_size_um)
        points_count = self._measurement_points_count(measurement)
        if points_count > 0 and measurement.length_um:
            return float(measurement.length_um) / points_count
        return None

    def _measurement_avg_size_text(self, measurement: StoredMeasurement) -> str:
        value = self._measurement_avg_size_um(measurement)
        return f"{value:.2f}" if value is not None else "—"

    def _measurement_result_um(self, measurement: StoredMeasurement) -> float:
        if measurement.measurement_type == "intersects":
            grain_size_um = getattr(measurement, "grain_size_um", None)
            if grain_size_um is not None:
                return float(grain_size_um)
            points_count = self._measurement_points_count(measurement)
            if points_count > 0 and self.calibration:
                return float(measurement.length_um) / points_count
        return float(measurement.length_um)

    def _restore_default_interaction_modes(self) -> None:
        self.ref_viewer.set_move_overlay_enabled(self.move_overlay_action.isChecked())
        self.ref_viewer.set_view_mode("navigate")
        self.sample_viewer.set_view_mode("navigate")
        self.sample_viewer.enable_point_selection(False)
        self.sample_viewer.disconnect_point()
        self.sample_viewer.disconnect_hover()
        self.sample_viewer.clear_measurement_preview()

    def _on_enhancement_changed(self) -> None:
        """Handle changes to sample enhancement sliders with debouncing."""
        brightness = float(self.brightness_slider.value())
        contrast = float(self.contrast_slider.value()) / 100.0
        saturation = float(self.saturation_slider.value()) / 100.0
        
        # Update value labels immediately for feedback
        self.brightness_value_label.setText(f"{int(brightness)}")
        self.contrast_value_label.setText(f"{contrast:.2f}")
        self.saturation_value_label.setText(f"{saturation:.2f}")
        
        # Queue enhancement and restart debounce timer
        self._sample_enhancement_pending = (brightness, contrast, saturation)
        self._sample_enhancement_timer.start(100)  # 100ms debounce

    def _apply_sample_enhancement(self) -> None:
        """Apply pending sample viewer enhancement."""
        if self._sample_enhancement_pending:
            brightness, contrast, saturation = self._sample_enhancement_pending
            self.sample_viewer.set_enhancement(brightness, contrast, saturation)

    def _on_ref_enhancement_changed(self) -> None:
        """Handle changes to reference enhancement sliders with debouncing."""
        brightness = float(self.ref_brightness_slider.value())
        contrast = float(self.ref_contrast_slider.value()) / 100.0
        saturation = float(self.ref_saturation_slider.value()) / 100.0
        
        # Update value labels immediately for feedback
        self.ref_brightness_value_label.setText(f"{int(brightness)}")
        self.ref_contrast_value_label.setText(f"{contrast:.2f}")
        self.ref_saturation_value_label.setText(f"{saturation:.2f}")
        
        # Queue enhancement and restart debounce timer
        self._ref_enhancement_pending = (brightness, contrast, saturation)
        self._ref_enhancement_timer.start(100)  # 100ms debounce

    def _apply_ref_enhancement(self) -> None:
        """Apply pending reference viewer enhancement."""
        if self._ref_enhancement_pending:
            brightness, contrast, saturation = self._ref_enhancement_pending
            self.ref_viewer.set_enhancement(brightness, contrast, saturation)

    def _on_reset_enhancements(self) -> None:
        """Reset all enhancements to defaults."""
        self.brightness_slider.setValue(0)
        self.contrast_slider.setValue(100)
        self.saturation_slider.setValue(100)
        # apply defaults to sample viewer
        self.sample_viewer.set_enhancement(0.0, 1.0, 1.0)

    def _on_reset_ref_enhancements(self) -> None:
        """Reset reference enhancement sliders to defaults."""
        self.ref_brightness_slider.setValue(0)
        self.ref_contrast_slider.setValue(100)
        self.ref_saturation_slider.setValue(100)
        # apply defaults to reference viewer
        self.ref_viewer.set_enhancement(0.0, 1.0, 1.0)

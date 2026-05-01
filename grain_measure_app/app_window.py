from PySide6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QToolBar,
    QFileDialog, QMessageBox, QTableWidget, QTableWidgetItem, QHeaderView, QInputDialog,
    QSlider, QCheckBox, QSizePolicy, QAbstractItemView, QToolButton, QLabel, QPushButton
)
from PySide6.QtGui import QAction
from PySide6.QtCore import Qt, QSignalBlocker
from functools import partial
import math

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
        self._selected_measurement_id: int | None = None
        self.overlay_scale: float = 1.0

        self._init_ui()

    def _init_ui(self) -> None:
        toolbar = QToolBar("Main")
        self.addToolBar(toolbar)

        load_ref_action = QAction("Load reference", self)
        load_ref_action.triggered.connect(self.load_reference_image)
        toolbar.addAction(load_ref_action)

        load_sample_action = QAction("Load sample", self)
        load_sample_action.triggered.connect(self.load_sample_image)
        toolbar.addAction(load_sample_action)

        calibrate_action = QAction("Calibrate", self)
        calibrate_action.triggered.connect(self.start_calibration)
        toolbar.addAction(calibrate_action)

        scale_sample_action = QAction("Scale sample", self)
        scale_sample_action.setCheckable(True)
        scale_sample_action.toggled.connect(self._on_scale_sample_toggled)
        self.scale_sample_action = scale_sample_action
        toolbar.addAction(scale_sample_action)

        measure_action = QAction("Measure", self)
        measure_action.setCheckable(True)
        measure_action.toggled.connect(self._on_measure_toggled)
        self.measure_action = measure_action
        toolbar.addAction(measure_action)

        move_overlay_action = QAction("Move overlay", self)
        move_overlay_action.setCheckable(True)
        move_overlay_action.toggled.connect(self._on_move_overlay_toggled)
        self.move_overlay_action = move_overlay_action
        toolbar.addAction(move_overlay_action)

        export_action = QAction("Export CSV", self)
        export_action.triggered.connect(self.export_csv)
        toolbar.addAction(export_action)

        central = QWidget()
        self.setCentralWidget(central)

        main_layout = QVBoxLayout(central)

        # Top: viewers with separate control columns
        viewers_and_controls_layout = QHBoxLayout()

        # Left column: reference viewer
        left_column = QVBoxLayout()
        self.ref_viewer = ImageViewer()
        left_column.addWidget(self.ref_viewer)
        # Reference enhancement controls (under reference viewer)
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
        left_column.addLayout(ref_enh_layout)
        viewers_and_controls_layout.addLayout(left_column, 3)

        # Right column: sample viewer + enhancement controls
        right_column = QVBoxLayout()
        self.sample_viewer = ImageViewer()
        right_column.addWidget(self.sample_viewer, 1)

        # enhancement controls (brightness, contrast, saturation) - on the right under sample
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

        right_column.addLayout(enhancement_layout)
        viewers_and_controls_layout.addLayout(right_column, 3)

        main_layout.addLayout(viewers_and_controls_layout, 8)

        # overlay controls (place under reference viewer on the left)
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
        left_column.addLayout(overlay_layout)

        # connect overlay scale changes (update table/display when user scales)
        try:
            self.ref_viewer.connect_overlay_scale_changed(self._on_overlay_scale_changed)
        except Exception:
            pass

        # Statistics panel above measurements table (place under sample viewer on right)
        stats_layout = QHBoxLayout()
        self.stats_label = QLabel("Measurements: 0 | Avg: — | Std Dev: —")
        stats_layout.addWidget(self.stats_label)
        stats_layout.addStretch()
        # Toggle to show measurement labels on sample image
        self.show_labels_checkbox = QCheckBox("Show measurement labels")
        self.show_labels_checkbox.setChecked(False)
        self.show_labels_checkbox.stateChanged.connect(lambda s: self._on_show_measurement_labels(bool(s)))
        stats_layout.addWidget(self.show_labels_checkbox)
        stats_layout.addStretch()
        right_column.addLayout(stats_layout)

        # Measurement table (more compact: remove Type and Accepted columns) under sample viewer
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["ID", "Pixels", "Length µm (10⁻⁶m)", "Delete"])
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
        right_column.addWidget(self.table, 1)

    def load_reference_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open reference image")
        if not path:
            return
        self.ref_viewer.load_image(path)
        # if overlay mode active and sample loaded, set overlay
        if self.overlay_checkbox.isChecked() and self.sample_viewer._image_path:
            self.ref_viewer.set_overlay_image(self.sample_viewer._image_path)
            self.ref_viewer.set_move_overlay_enabled(self.move_overlay_action.isChecked())

    def load_sample_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open sample image")
        if not path:
            return
        self.sample_viewer.load_image(path)
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
            self.sample_viewer.set_view_mode("pick")
            self.statusBar().showMessage("Measurement: click two points on the sample image")
        else:
            self._measurement_points = []
            self.sample_viewer.clear_measurement_preview()
            self.sample_viewer.disconnect_point()
            self.sample_viewer.enable_point_selection(False)
            self.sample_viewer.set_view_mode("navigate")
            self.statusBar().showMessage("Measurement mode off")

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

    def export_csv(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Export measurements", filter="CSV Files (*.csv)")
        if not path:
            return
        try:
            export_csv.export_measurements_csv(path, self.measurements.get_all(), self.calibration, overlay_scale=self.overlay_scale)
            QMessageBox.information(self, "Export", f"Exported CSV to:\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to export CSV:\n{e}")

    def refresh_table(self) -> None:
        current_selection = self._selected_measurement_id
        blocker = QSignalBlocker(self.table)
        rows = self.measurements.count()
        self.table.setRowCount(rows)
        for i, m in enumerate(self.measurements.get_all()):
            self.table.setItem(i, 0, QTableWidgetItem(str(m.id)))
            # display pixel length and µm adjusted by current overlay scale
            display_pixels = m.pixel_length * float(self.overlay_scale)
            display_um = display_pixels * (self.calibration.um_per_pixel if self.calibration else (m.length_um / m.pixel_length if m.pixel_length else 0.0))
            self.table.setItem(i, 1, QTableWidgetItem(f"{display_pixels:.2f}"))
            self.table.setItem(i, 2, QTableWidgetItem(f"{display_um:.2f}"))
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
            self.table.setCellWidget(i, 3, delete_button)

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
            self.stats_label.setText("Measurements: 0 | Avg: — | Std Dev: —")
            return
        
        # Calculate average and standard deviation of length_um
        lengths = [m.length_um for m in measurements]
        avg_length = sum(lengths) / count
        
        # Calculate standard deviation
        variance = sum((x - avg_length) ** 2 for x in lengths) / count
        std_dev = variance ** 0.5
        
        # Coefficient of variation (std dev as % of mean) for relative spread
        cv_percent = (std_dev / avg_length * 100) if avg_length > 0 else 0
        
        self.stats_label.setText(
            f"Measurements: {count} | Avg: {avg_length:.2f} µm | Std Dev: {std_dev:.2f} µm ({cv_percent:.1f}%)"
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
        lines = [MeasurementLine(m.x1, m.y1, m.x2, m.y2, m.id) for m in self.measurements.get_all()]
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
            display_pixels = m.pixel_length * float(self.overlay_scale)
            display_um = display_pixels * self.calibration.um_per_pixel
            mapping[m.id] = f"{display_um:.2f} µm"
        self.sample_viewer.set_measurement_label_texts(mapping)

    def _restore_default_interaction_modes(self) -> None:
        self.ref_viewer.set_move_overlay_enabled(self.move_overlay_action.isChecked())
        self.ref_viewer.set_view_mode("navigate")
        self.sample_viewer.set_view_mode("navigate")
        self.sample_viewer.enable_point_selection(False)
        self.sample_viewer.disconnect_point()
        self.sample_viewer.disconnect_hover()
        self.sample_viewer.clear_measurement_preview()

    def _on_enhancement_changed(self) -> None:
        """Handle changes to enhancement sliders."""
        brightness = float(self.brightness_slider.value())
        contrast = float(self.contrast_slider.value()) / 100.0
        saturation = float(self.saturation_slider.value()) / 100.0
        
        # Update value labels
        self.brightness_value_label.setText(f"{int(brightness)}")
        self.contrast_value_label.setText(f"{contrast:.2f}")
        self.saturation_value_label.setText(f"{saturation:.2f}")
        
        # Apply enhancements to sample viewer only
        self.sample_viewer.set_enhancement(brightness, contrast, saturation)

    def _on_ref_enhancement_changed(self) -> None:
        """Handle changes to reference enhancement sliders."""
        brightness = float(self.ref_brightness_slider.value())
        contrast = float(self.ref_contrast_slider.value()) / 100.0
        saturation = float(self.ref_saturation_slider.value()) / 100.0
        
        # Update value labels
        self.ref_brightness_value_label.setText(f"{int(brightness)}")
        self.ref_contrast_value_label.setText(f"{contrast:.2f}")
        self.ref_saturation_value_label.setText(f"{saturation:.2f}")
        
        # Apply enhancements to reference viewer only
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

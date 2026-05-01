from PySide6.QtCore import QObject, QThread, Signal
from typing import Optional


class EnhancementWorker(QObject):
    """Worker thread for applying image enhancements without blocking the GUI."""
    
    finished = Signal()
    
    def __init__(self):
        super().__init__()
        self._pending_task: Optional[tuple[float, float, float]] = None
        self._is_processing = False
    
    def queue_enhancement(self, brightness: float, contrast: float, saturation: float) -> None:
        """Queue an enhancement task. The most recent request will be processed."""
        self._pending_task = (brightness, contrast, saturation)
        if not self._is_processing:
            self.process_pending()
    
    def process_pending(self) -> None:
        """Process the pending enhancement task if one exists."""
        if self._pending_task is None:
            self._is_processing = False
            self.finished.emit()
            return
        
        self._is_processing = True
        brightness, contrast, saturation = self._pending_task
        self._pending_task = None
        
        # Emit signal with the enhancement values to be applied
        self.finished.emit()


class EnhancementThreadPool:
    """Simple pool to manage enhancement workers for reference and sample viewers."""
    
    def __init__(self):
        self.ref_thread: Optional[QThread] = None
        self.ref_worker: Optional[EnhancementWorker] = None
        self.sample_thread: Optional[QThread] = None
        self.sample_worker: Optional[EnhancementWorker] = None
    
    def init_ref_worker(self, on_done_callback):
        """Initialize reference viewer enhancement worker."""
        if self.ref_thread is None:
            self.ref_thread = QThread()
            self.ref_worker = EnhancementWorker()
            self.ref_worker.moveToThread(self.ref_thread)
            self.ref_worker.finished.connect(on_done_callback)
            self.ref_thread.start()
        return self.ref_worker
    
    def init_sample_worker(self, on_done_callback):
        """Initialize sample viewer enhancement worker."""
        if self.sample_thread is None:
            self.sample_thread = QThread()
            self.sample_worker = EnhancementWorker()
            self.sample_worker.moveToThread(self.sample_thread)
            self.sample_worker.finished.connect(on_done_callback)
            self.sample_thread.start()
        return self.sample_worker
    
    def cleanup(self):
        """Clean up threads."""
        if self.ref_thread:
            self.ref_thread.quit()
            self.ref_thread.wait()
        if self.sample_thread:
            self.sample_thread.quit()
            self.sample_thread.wait()

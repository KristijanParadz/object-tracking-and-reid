from typing import Tuple


class Config:
    # Define your constants as class attributes
    EMBEDDING_UPDATE_INTERVAL: int = 60
    SKIP_INTERVAL: int = 5
    RESIZE_SHAPE: Tuple[int, int] = (640, 360)
    YOLO_MODEL_PATH: str = 'yolov8n.pt'
    SIMILARITY_THRESHOLD: float = 0.3
    RESIZE_SHAPE_FOR_FRONTEND: Tuple[int, int] = (640, 360)
    FULLSCREEN_CAMERA: str | None = None

    def __new__(cls, *args, **kwargs):
        raise TypeError("Config is a static class and cannot be instantiated")

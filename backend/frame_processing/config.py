from typing import Tuple


class Config:
    # Define your constants as class attributes
    EMBEDDING_UPDATE_INTERVAL: int = 60
    SKIP_INTERVAL: int = 5
    RESIZE_SHAPE: Tuple[int, int] = (640, 360)
    SCALE: float = 3  # when changing resize shape this has to be adjusted too
    YOLO_MODEL_PATH: str = 'yolo12n.pt'
    SIMILARITY_THRESHOLD: float = 0.3
    EPIPOLAR_THRESHOLD = 100
    EMBEDDING_ALPHA = 0.7

    # dynamic values
    RESIZE_SHAPE_FOR_FRONTEND: Tuple[int, int] = (640, 360)
    FULLSCREEN_CAMERA: str | None = None
    INTRINSIC_FRAME_REQUESTED: bool = False
    INTRINSIC_FRAME_NUMBER_TO_SAVE: int = -1
    EXTRINSIC_FRAME_REQUESTED: bool = False
    EXTRINSIC_FRAME_NUMBER_TO_SAVE: int = -1

    def __new__(cls, *args, **kwargs):
        raise TypeError("Config is a static class and cannot be instantiated")

from __future__ import annotations

from typing import Optional, Tuple


class Config:
    """
    Centralized configuration values used across the pipeline.

    Notes:
    - This is a static holder; instantiation is blocked on purpose.
    - Thresholds that participate in float math are declared as floats.
    """

    # ---------- Tracking / Embeddings ----------
    EMBEDDING_UPDATE_INTERVAL: int = 60          # frames between embedding updates
    SKIP_INTERVAL: int = 5                       # process every Nth frame
    SIMILARITY_THRESHOLD: float = 0.30           # cosine similarity cutoff
    EMBEDDING_ALPHA: float = 0.50                # running blend for embeddings

    # ---------- Geometry / Epipolar ----------
    EPIPOLAR_THRESHOLD: float = 100.0            # px distance to epipolar line

    # ---------- Image / Resizing ----------
    RESIZE_SHAPE: Tuple[int, int] = (640, 360)   # (width, height)
    SCALE: float = 3.0                            # keep in sync with RESIZE_SHAPE

    # ---------- Models ----------
    YOLO_MODEL_PATH: str = "yolo12n.pt"

    # ---------- Calibration (chessboard) ----------
    # For calibration/testing; pattern is number of inner corners.
    CHESSBOARD_PATTERN_SIZE: Tuple[int, int] = (
        7, 7)  # e.g., 9x6 in some setups
    CHESSBOARD_SQUARE_SIZE: float = 1.0                 # in chosen length units

    # ---------- Frontend / Runtime toggles (dynamic at runtime) ----------
    RESIZE_SHAPE_FOR_FRONTEND: Tuple[int, int] = (640, 360)
    FULLSCREEN_CAMERA: Optional[str] = None

    INTRINSIC_FRAME_REQUESTED: bool = False
    INTRINSIC_FRAME_NUMBER_TO_SAVE: int = -1

    EXTRINSIC_FRAME_REQUESTED: bool = False
    EXTRINSIC_FRAME_NUMBER_TO_SAVE: int = -1

    def __new__(cls, *args, **kwargs) -> "Config":
        # Prevent accidental instantiation: this is a static config holder.
        raise TypeError("Config is a static class and cannot be instantiated")

from __future__ import annotations
from typing import Protocol, runtime_checkable
from typing import Protocol, runtime_checkable, Optional, Tuple
from numpy.typing import NDArray
import numpy as np


@runtime_checkable
class ReIDModelProtocol(Protocol):
    def get_embedding(
        self, image_bgr: NDArray[np.uint8]) -> NDArray[np.float32]: ...


@runtime_checkable
class GlobalIDManagerProtocol(Protocol):
    def match_or_create(
        self,
        embedding: NDArray[np.float32],
        class_id: int,
        camera_id: str,
        bbox_center,               # your Point type; keep untyped here to avoid imports
        frame_number: int,
    ) -> int: ...

    def update_position(
        self, global_id: int, camera_id: str, bbox_center, frame_number: int
    ) -> None: ...

    def update_embedding(
        self, global_id: int, new_embedding: NDArray[np.float32], alpha: Optional[float] = None
    ) -> None: ...

    def get_color(self, global_id: int) -> Tuple[int, int, int]: ...

    def get_triangulated_point(
        self, global_id: int) -> Optional[NDArray[np.float64]]: ...

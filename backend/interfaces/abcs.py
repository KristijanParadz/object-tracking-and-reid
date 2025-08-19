# interfaces/bases.py
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

# Small, shared lifecycle ABC


class StreamerBase(ABC):
    """Minimal lifecycle for streaming/calibration components."""
    running: bool

    @abstractmethod
    async def start(self) -> None:
        """Begin the async streaming loop."""
        ...

    def stop(self) -> None:
        """Default stop behavior (can be overridden)."""
        self.running = False

    @abstractmethod
    def _cleanup(self) -> None:
        """Release resources (captures/buffers)."""
        ...


# Intrinsic-specific ABC
class IntrinsicCalibratorBase(StreamerBase):
    @abstractmethod
    def get_saved_images_base64(self) -> List[str]:
        ...

    @abstractmethod
    def calibrate_intrinsic_from_folder(
        self,
        pattern_size: Tuple[int, int],
        square_size: float,
    ) -> dict:
        ...


# Extrinsic-specific ABC
class ExtrinsicCalibratorBase(StreamerBase):
    @abstractmethod
    def get_saved_images_base64(self) -> Dict[str, List[str]]:
        ...

    @abstractmethod
    def calibrate_all_extrinsics(
        self,
        intrinsics: Dict[int, Dict[str, object]],
        pattern_size: Tuple[int, int],
        square_size: float,
    ) -> Dict[int, Optional[Dict[str, List[List[float]]]]]:
        ...

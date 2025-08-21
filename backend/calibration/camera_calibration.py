from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


def _as_float_array(value: Any) -> FloatArray:
    """Convert input to a NumPy float64 array."""
    return np.array(value, dtype=float)


@dataclass(frozen=True)
class CalibrationParameters:
    """
    Container for a single camera's calibration parameters.

    Attributes:
        K: Intrinsic matrix (3x3).
        dist_coef: Distortion coefficients (1x5).
        R: Rotation matrix (3x3).
        t: Translation vector (3x1).
    """
    K: FloatArray
    dist_coef: FloatArray
    R: FloatArray
    t: FloatArray

    def __post_init__(self) -> None:
        # Shape validations
        if self.K.shape != (3, 3):
            raise ValueError(f"Expected K shape (3, 3), got {self.K.shape}")
        if self.R.shape != (3, 3):
            raise ValueError(f"Expected R shape (3, 3), got {self.R.shape}")
        if self.t.shape not in {(3, 1), (3,)}:
            raise ValueError(
                f"Expected t shape (3, 1) or (3,), got {self.t.shape}")


class CameraCalibration:
    """
    Loader/validator for multiple cameras' calibration data.
    """

    cameras: Dict[str, CalibrationParameters]

    def __init__(self, calib_list: Iterable[Mapping[str, Any]]) -> None:
        """
        Load calibration data from an iterable of camera dictionaries.

        Each element must include keys: "index", "K", "distCoef" (or "dist_coef"),
        "R", and "t".

        Args:
            calib_list: Iterable of per-camera dicts.
        """
        self.cameras = {}
        for cam in calib_list:
            name = f"Camera {cam['index']}"
            dist = cam.get("dist_coef", cam.get("distCoef"))
            params = CalibrationParameters(
                K=_as_float_array(cam["K"]),
                dist_coef=_as_float_array(dist),
                R=_as_float_array(cam["R"]),
                t=_as_float_array(cam["t"]).reshape(3, 1),
            )
            self.cameras[name] = params

    @staticmethod
    def validate_calibration(
        calib_data: Mapping[str, Mapping[str, Any]]
    ) -> Dict[str, CalibrationParameters]:
        """
        Ensure calibration entries are convertible to the correct types/shapes.

        Args:
            calib_data: Mapping of camera id -> dict with keys "K",
                        "distCoef"/"dist_coef", "R", "t".

        Returns:
            Dict mapping camera id to validated CalibrationParameters.

        Raises:
            ValueError: If required keys are missing or shapes are invalid.
        """
        validated: Dict[str, CalibrationParameters] = {}
        for cam_id, data in calib_data.items():
            try:
                dist = data.get("dist_coef", data.get("distCoef"))
                params = CalibrationParameters(
                    K=_as_float_array(data["K"]),
                    dist_coef=_as_float_array(dist),
                    R=_as_float_array(data["R"]),
                    t=_as_float_array(data["t"]).reshape(3, 1),
                )
                validated[cam_id] = params
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid calibration data for camera '{cam_id}': {exc}"
                ) from exc
        return validated

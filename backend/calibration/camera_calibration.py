import numpy as np


class CameraCalibration:
    def __init__(self, calib_list):
        """ Load calibration data from a list of camera dictionaries. """
        self.cameras = {}
        for cam in calib_list:
            name = f"Camera {cam['index']}"
            self.cameras[name] = {
                "K": np.array(cam["K"], dtype=float),
                "distCoef": np.array(cam["distCoef"], dtype=float),
                "R": np.array(cam["R"], dtype=float),
                "t": np.array(cam["t"], dtype=float).reshape(3, 1)
            }

    def validate_calibration(self, calib_data):
        """Ensures calibration data has numpy arrays with correct shapes."""
        validated_data = {}
        for cam_id, data in calib_data.items():
            try:
                validated_data[cam_id] = {
                    "K": np.array(data["K"], dtype=float),
                    "distCoef": np.array(data["distCoef"], dtype=float),
                    "R": np.array(data["R"], dtype=float),
                    "t": np.array(data["t"], dtype=float).reshape(3, 1)
                }
                # Basic shape checks
                assert validated_data[cam_id]["K"].shape == (3, 3)
                assert validated_data[cam_id]["R"].shape == (3, 3)
                assert validated_data[cam_id]["t"].shape == (3, 1)
            except (KeyError, TypeError, AssertionError) as e:
                raise ValueError(
                    f"Invalid calibration data format for camera {cam_id}: {e}")
        return validated_data


class CalibrationParameters:
    def __init__(self, K, distCoef, R, t):
        self.K = np.array(K)
        self.distCoef = np.array(distCoef)
        self.R = np.array(R)
        self.t = np.array(t)

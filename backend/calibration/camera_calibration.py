import numpy as np
import json


class CameraCalibration:
    def __init__(self, calib_file):
        """ Load calibration data from JSON. """
        with open(calib_file, 'r') as f:
            self.calib_data = json.load(f)

        self.cameras = {}
        for cam in self.calib_data["cameras"]:
            name = f"{cam['type']}_{cam['name']}"
            self.cameras[name] = {
                "K": np.array(cam["K"]),
                "distCoef": np.array(cam["distCoef"]),
                "R": np.array(cam["R"]),
                "t": np.array(cam["t"]),
            }


class CalibrationParameters:
    def __init__(self, K, distCoef, R, t):
        self.K = K
        self.distCoef = distCoef
        self.R = R
        self.t = t

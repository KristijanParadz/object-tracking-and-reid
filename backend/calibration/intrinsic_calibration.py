import cv2
import base64
import glob
import asyncio
import os
import shutil
from collections import deque
from typing import Deque, Tuple
from frame_processing.config import Config
import re
import numpy as np


class IntrinsicCameraStreamer:
    def __init__(self, sio, camera_index: int, output_dir='calibration/intrinsic_images'):
        self.sio = sio
        self.camera_index = camera_index
        self.output_dir = output_dir
        self.frame_buffer: Deque[Tuple[int, any]] = deque(maxlen=30)
        self.frame_counter = 0
        self.frames_saved = 0
        self.running = False
        self.cap = None

    def encode_frame_to_base64(self, frame):
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            return None
        return base64.b64encode(buffer).decode('utf-8')

    async def start(self):
        # Clear image output directory
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)
            print(f"Cleared folder: {self.output_dir}")

        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            print(f"Error: Cannot open camera index {self.camera_index}")
            return

        self.running = True
        os.makedirs(self.output_dir, exist_ok=True)

        while self.running:
            if self.frames_saved == 10:
                self.stop()

            ret, frame = self.cap.read()
            if not ret:
                print("Warning: Failed to grab frame")
                continue

            self.frame_buffer.append((self.frame_counter, frame.copy()))

            encoded = self.encode_frame_to_base64(frame)
            if encoded:
                await self.sio.emit('live-feed-intrinsic', {
                    'image': encoded,
                    'frame_number': self.frame_counter,
                    'frames_saved': self.frames_saved
                })

            if Config.INTRINSIC_FRAME_REQUESTED:
                Config.INTRINSIC_FRAME_REQUESTED = False

                found = False
                for frame_number, saved_frame in self.frame_buffer:
                    if frame_number == Config.INTRINSIC_FRAME_NUMBER_TO_SAVE:
                        save_path = f"{self.output_dir}/frame_{frame_number}.jpg"
                        cv2.imwrite(save_path, saved_frame)
                        self.frames_saved += 1
                        print(f"Saved frame {frame_number} -> {save_path}")
                        found = True
                        break

                if not found and self.frame_buffer:
                    fallback_fn, fallback_frame = self.frame_buffer[0]
                    save_path = f"{self.output_dir}/frame_{fallback_fn}.jpg"
                    cv2.imwrite(save_path, fallback_frame)
                    self.frames_saved += 1
                    print(f"Warning: Requested frame {Config.INTRINSIC_FRAME_NUMBER_TO_SAVE} not in buffer. "
                          f"Saved fallback frame {fallback_fn} -> {save_path}")

            self.frame_counter += 1
            await asyncio.sleep(0.05)  # Use asyncio sleep in async context

        # Clean up after stopping
        self._cleanup()

    def stop(self):
        """Call this from outside to stop the feed and clean everything up."""
        self.running = False

    def _cleanup(self):
        if self.cap and self.cap.isOpened():
            self.cap.release()
        self.frame_buffer.clear()

    def get_saved_images_base64(self):
        """
        Returns a list of base64-encoded strings of all images in self.output_dir,
        sorted by the frame number in the filename.
        """
        base64_images = []
        if not os.path.exists(self.output_dir):
            print(f"Directory {self.output_dir} does not exist.")
            return base64_images

        def extract_frame_number(path):
            match = re.search(r'frame_(\d+)\.jpg', os.path.basename(path))
            return int(match.group(1)) if match else -1

        image_paths = sorted(
            glob.glob(os.path.join(self.output_dir, '*.jpg')),
            key=extract_frame_number
        )

        for img_path in image_paths:
            try:
                with open(img_path, 'rb') as image_file:
                    encoded = base64.b64encode(
                        image_file.read()).decode('utf-8')
                    base64_images.append(encoded)
            except Exception as e:
                print(f"Failed to encode image {img_path}: {e}")
        return base64_images

    def calibrate_intrinsic_from_folder(self, pattern_size=(9, 6), square_size=1.0):
        """
        Performs intrinsic camera calibration from chessboard images in a folder.

        Args:
            image_dir (str): Path to the folder containing calibration images.
            pattern_size (tuple): Number of inner corners per chessboard row and column (cols, rows).
            square_size (float): Size of a square on the chessboard (in user-defined units, e.g., mm).

        Returns:
            dict: Calibration results containing camera matrix, distortion coefficients, etc.
        """
        # Termination criteria for sub-pixel refinement
        criteria = (cv2.TERM_CRITERIA_EPS +
                    cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # Prepare object points like (0,0,0), (1,0,0), ..., (8,5,0) multiplied by square size
        objp = np.zeros((pattern_size[1] * pattern_size[0], 3), np.float32)
        objp[:, :2] = np.mgrid[0:pattern_size[0],
                               0:pattern_size[1]].T.reshape(-1, 2)
        objp *= square_size

        objpoints = []  # 3D points in real-world space
        imgpoints = []  # 2D points in image plane

        test_output_dir = "calibration/example_images"

        images = sorted(glob.glob(os.path.join(test_output_dir, '*.jpg')))
        if not images:
            raise FileNotFoundError(
                f"No images found in directory: {test_output_dir}")

        image_shape = None
        for fname in images:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Save image size for calibration
            if image_shape is None:
                image_shape = gray.shape[::-1]

            # Find the chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)

            if ret:
                objpoints.append(objp)
                corners2 = cv2.cornerSubPix(
                    gray, corners, (11, 11), (-1, -1), criteria)
                imgpoints.append(corners2)
            else:
                print(f"Warning: Chessboard not found in {fname}")

        if not objpoints or not imgpoints:
            raise RuntimeError(
                "No valid chessboard corners found in any image.")

        # Calibrate camera
        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, image_shape, None, None
        )

        return {
            "K": camera_matrix.tolist(),
            "distCoef": dist_coeffs.tolist()[0],
        }

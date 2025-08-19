from __future__ import annotations

import asyncio
import base64
import os
import shutil
from collections import deque
from typing import Deque, List, Optional, Tuple

import cv2
import numpy as np
import socketio

from frame_processing.config import Config
from calibration.utils import make_frame_encoder, make_frame_number_extractor
from interfaces.abcs import IntrinsicCalibratorBase


class IntrinsicCameraStreamer(IntrinsicCalibratorBase):
    """
    Streams frames from a camera, runs chessboard detection periodically,
    allows saving specific frames to disk, and can compute intrinsic calibration
    from the saved images.
    """

    sio: socketio.AsyncServer
    camera_index: int
    output_dir: str
    frame_buffer: Deque[Tuple[int, np.ndarray]]
    frame_counter: int
    frames_saved: int
    running: bool
    cap: Optional[cv2.VideoCapture]

    # Detection state (emitted to frontend)
    is_detected: bool
    corners_found: int
    reason: str

    def __init__(
        self,
        sio: socketio.AsyncServer,
        camera_index: int,
        output_dir: str = "calibration/intrinsic_images",
    ) -> None:
        self.sio = sio
        self.camera_index = camera_index
        self.output_dir = output_dir
        self.frame_buffer = deque(maxlen=30)
        self.frame_counter = 0
        self.frames_saved = 0
        self.running = False
        self.cap = None

        # Detection state
        self.is_detected = False
        self.corners_found = 0
        self.reason = "not_run"

    # ---------------- Detection ----------------

    def _run_detection(self, frame: np.ndarray) -> None:
        """Run chessboard detection and update detection state."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flags = cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY

        found, corners = cv2.findChessboardCornersSB(
            gray, Config.CHESSBOARD_PATTERN_SIZE, flags=flags
        )
        if found and corners is not None:
            self.is_detected = True
            self.corners_found = int(corners.shape[0])
            self.reason = "found"
        else:
            self.is_detected = False
            self.corners_found = 0
            self.reason = "not_found"

    # ---------------- Streaming ----------------

    async def start(self) -> None:
        """
        Start streaming frames:
        - clears output directory,
        - reads frames,
        - runs detection every 5th frame,
        - serves live images to frontend,
        - saves requested frames from a rolling buffer.
        """
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

            ok, frame = self.cap.read()
            if not ok:
                print("Warning: Failed to grab frame")
                await asyncio.sleep(0.05)
                continue

            # Store a copy in the rolling buffer: (frame_number, frame)
            self.frame_buffer.append((self.frame_counter, frame.copy()))

            # Run detection every 5th frame
            if self.frame_counter % 5 == 0:
                try:
                    self._run_detection(frame)
                except Exception as exc:  # be robust to OpenCV edge cases
                    self.is_detected = False
                    self.corners_found = 0
                    self.reason = f"error:{type(exc).__name__}"

            # Emit current frame to frontend
            encode_frame_to_base64 = make_frame_encoder()

            encoded = encode_frame_to_base64(frame)
            if encoded is not None:
                await self.sio.emit(
                    "live-feed-intrinsic",
                    {
                        "image": encoded,
                        "frame_number": self.frame_counter,
                        "frames_saved": self.frames_saved,
                        "is_detected": self.is_detected,
                        "corners_found": self.corners_found,
                        "reason": self.reason,
                    },
                )

            # Handle save request from Config (static runtime flags)
            if Config.INTRINSIC_FRAME_REQUESTED:
                Config.INTRINSIC_FRAME_REQUESTED = False

                found_in_buffer = False
                for frame_number, saved_frame in self.frame_buffer:
                    if frame_number == Config.INTRINSIC_FRAME_NUMBER_TO_SAVE:
                        save_path = f"{self.output_dir}/frame_{frame_number}.jpg"
                        cv2.imwrite(save_path, saved_frame)
                        self.frames_saved += 1
                        print(f"Saved frame {frame_number} -> {save_path}")
                        found_in_buffer = True
                        break

                # Fallback: save the oldest buffered frame if the requested one is gone
                if not found_in_buffer and self.frame_buffer:
                    fallback_fn, fallback_frame = self.frame_buffer[0]
                    save_path = f"{self.output_dir}/frame_{fallback_fn}.jpg"
                    cv2.imwrite(save_path, fallback_frame)
                    self.frames_saved += 1
                    print(
                        "Warning: Requested frame "
                        f"{Config.INTRINSIC_FRAME_NUMBER_TO_SAVE} not in buffer. "
                        f"Saved fallback frame {fallback_fn} -> {save_path}"
                    )

            self.frame_counter += 1

        self._cleanup()

    def stop(self) -> None:
        """Signal the streaming loop to stop."""
        self.running = False

    def _cleanup(self) -> None:
        """Release camera and clear buffers."""
        if self.cap and self.cap.isOpened():
            self.cap.release()
        self.frame_buffer.clear()

    # ---------------- Utilities ----------------

    def get_saved_images_base64(self) -> List[str]:
        """
        Return base64 strings of all saved images in `output_dir`,
        sorted by frame number in filename.
        """
        base64_images: List[str] = []
        if not os.path.exists(self.output_dir):
            print(f"Directory {self.output_dir} does not exist.")
            return base64_images

        extract_frame_number = make_frame_number_extractor()

        import glob  # local import to reduce module scope
        image_paths = sorted(
            glob.glob(os.path.join(self.output_dir, "*.jpg")), key=extract_frame_number
        )

        for img_path in image_paths:
            try:
                with open(img_path, "rb") as f:
                    encoded = base64.b64encode(f.read()).decode("utf-8")
                    base64_images.append(encoded)
            except Exception as exc:
                print(f"Failed to encode image {img_path}: {exc}")
        return base64_images

    def calibrate_intrinsic_from_folder(
        self,
        pattern_size: Tuple[int, int] = Config.CHESSBOARD_PATTERN_SIZE,
        square_size: float = Config.CHESSBOARD_SQUARE_SIZE,
    ) -> dict:
        """
        Compute intrinsic calibration from chessboard images in `output_dir`.

        Args:
            pattern_size: (cols, rows) inner-corner count per chessboard.
            square_size: Physical size of one square (user units, e.g., mm).

        Returns:
            dict with:
              - "K": camera matrix (3x3) as nested list
              - "dist_coef": distortion coefficients as flat list
            (You can also include 'image_size' if desired.)
        """
        # Termination criteria for sub-pixel refinement
        criteria = (cv2.TERM_CRITERIA_EPS +
                    cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)

        # Prepare object points grid, scaled by square_size
        objp = np.zeros((pattern_size[1] * pattern_size[0], 3), np.float32)
        objp[:, :2] = np.mgrid[0: pattern_size[0],
                               0: pattern_size[1]].T.reshape(-1, 2)
        objp *= float(square_size)

        objpoints: List[np.ndarray] = []  # 3D points in world coords
        imgpoints: List[np.ndarray] = []  # 2D points in image coords

        import glob  # local import to reduce module scope
        images = sorted(glob.glob(os.path.join(self.output_dir, "*.jpg")))
        if not images:
            raise FileNotFoundError(
                f"No images found in directory: {self.output_dir}")

        image_shape = None
        for fname in images:
            img = cv2.imread(fname)
            if img is None:
                print(f"Warning: Failed to read {fname}")
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if image_shape is None:
                image_shape = gray.shape[::-1]  # (width, height)

            # Find the chessboard corners (classic detector here; robust & fast)
            found, corners = cv2.findChessboardCorners(
                gray, pattern_size, None)
            if found:
                # Sub-pixel refinement
                corners_refined = cv2.cornerSubPix(
                    gray, corners, (11, 11), (-1, -1), criteria
                )
                objpoints.append(objp)
                imgpoints.append(corners_refined)
            else:
                print(f"Warning: Chessboard not found in {fname}")

        if not objpoints or not imgpoints:
            raise RuntimeError(
                "No valid chessboard corners found in any image.")

        # Calibrate camera
        ok, K, dist, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, image_shape, None, None
        )
        if not ok:
            print("Warning: cv2.calibrateCamera returned non-optimal solution.")

        # OpenCV returns distortion as (N,1); flatten to a 1D Python list
        dist_flat = dist.ravel().tolist()

        return {
            "K": K.tolist(),
            "dist_coef": dist_flat,  # <-- PEP-8 name; your loader accepts this
            # Optional extras you may find useful later:
            # "image_size": list(image_shape),
            # "rvecs": [r.ravel().tolist() for r in rvecs],
            # "tvecs": [t.ravel().tolist() for t in tvecs],
        }

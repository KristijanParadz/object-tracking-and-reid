from __future__ import annotations

import asyncio
import base64
import glob
import os
import re
import shutil
from collections import deque
from typing import Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np
import socketio

from frame_processing.config import Config


class ExtrinsicCameraStreamer:
    """
    - Streams frames from multiple cameras.
    - Periodically checks for a chessboard in all views (for operator feedback).
    - On request, saves synchronized frames from each camera into per-camera folders.
    - Computes *pairwise* extrinsics relative to a reference camera.
    """

    sio: socketio.AsyncServer
    camera_indexes: List[int]
    output_dir: str
    captures: Dict[int, cv2.VideoCapture]
    frame_buffers: Dict[int, Deque[Tuple[int, np.ndarray]]]
    frame_counter: int
    frames_saved: int
    running: bool

    # Live detection state (across all cameras)
    is_detected: bool
    _detection_stride: int
    _live_max_side: int

    def __init__(
        self,
        sio: socketio.AsyncServer,
        camera_indexes: List[int],
        output_dir: str = "calibration/extrinsic_images",
    ) -> None:
        self.sio = sio
        self.camera_indexes = camera_indexes
        self.output_dir = output_dir
        self.captures = {}
        self.frame_buffers = {idx: deque(maxlen=30) for idx in camera_indexes}
        self.frame_counter = 0
        self.frames_saved = 0
        self.running = False

        # Live detection parameters
        self.is_detected = False
        self._detection_stride = 5   # run detection every 5th frame
        self._live_max_side = 640    # downscale for speed

    # ---------------- Encoding / helpers ----------------

    @staticmethod
    def encode_frame_to_base64(frame: np.ndarray) -> Optional[str]:
        """Encode a frame to base64-encoded JPEG; return None on failure."""
        ok, buffer = cv2.imencode(".jpg", frame)
        if not ok:
            return None
        return base64.b64encode(buffer).decode("utf-8")

    @staticmethod
    def _downscale_keep_aspect(img: np.ndarray, max_side: int) -> np.ndarray:
        """Downscale image keeping aspect ratio so max(H, W) == max_side."""
        h, w = img.shape[:2]
        m = max(h, w)
        if m <= max_side:
            return img
        scale = max_side / float(m)
        new_w, new_h = int(w * scale), int(h * scale)
        return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    def _found_chessboard(self, frame_bgr: np.ndarray) -> bool:
        """Fast check on a single downsized frame for operator feedback."""
        small = self._downscale_keep_aspect(frame_bgr, self._live_max_side)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        flags = cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
        found, _ = cv2.findChessboardCornersSB(
            gray, Config.CHESSBOARD_PATTERN_SIZE, flags=flags
        )
        return bool(found)

    # ---------------- Streaming ----------------

    async def start(self) -> None:
        """
        Start multi-camera streaming, periodic detection, live emit, and on-demand
        frame saving into per-camera folders under `output_dir`.
        """
        # Reset output folder
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)
            print(f"Cleared folder: {self.output_dir}")
        os.makedirs(self.output_dir, exist_ok=True)

        # Open all captures and per-camera subfolders
        for idx in self.camera_indexes:
            cap = cv2.VideoCapture(idx)
            if not cap.isOpened():
                raise RuntimeError(f"Error: Cannot open camera index {idx}")
            self.captures[idx] = cap
            os.makedirs(os.path.join(self.output_dir,
                        f"camera{idx}"), exist_ok=True)

        self.running = True

        while self.running:
            images_base64: Dict[str, str] = {}
            current_frames: Dict[int, np.ndarray] = {}

            # Grab frames from all cameras
            for idx, cap in self.captures.items():
                ok, frame = cap.read()
                if not ok:
                    print(
                        f"Error: Failed to grab frame from camera{idx}. Stopping.")
                    self.stop()
                    return

                self.frame_buffers[idx].append(
                    (self.frame_counter, frame.copy()))
                current_frames[idx] = frame

                encoded = self.encode_frame_to_base64(frame)
                if encoded is None:
                    print(
                        f"Error: Failed to encode frame from camera{idx}. Stopping.")
                    self.stop()
                    return
                images_base64[f"camera{idx}"] = encoded

            # Periodic detection across all cameras
            if self.frame_counter % self._detection_stride == 0:
                try:
                    self.is_detected = all(
                        self._found_chessboard(frm) for frm in current_frames.values()
                    )
                except Exception:
                    # On any error, set false but keep the loop alive
                    self.is_detected = False

            # Emit live multi-camera bundle
            await self.sio.emit(
                "live-feed-extrinsic",
                {
                    "images": images_base64,
                    "frame_number": self.frame_counter,
                    "frames_saved": self.frames_saved,
                    "is_detected": self.is_detected,
                },
            )

            # Save-on-request logic using rolling buffers
            if Config.EXTRINSIC_FRAME_REQUESTED:
                Config.EXTRINSIC_FRAME_REQUESTED = False
                frame_to_save = Config.EXTRINSIC_FRAME_NUMBER_TO_SAVE

                for idx, buffer in self.frame_buffers.items():
                    found = False
                    for fnum, frm in buffer:
                        if fnum == frame_to_save:
                            path = os.path.join(
                                self.output_dir, f"camera{idx}", f"frame_{fnum}.jpg"
                            )
                            cv2.imwrite(path, frm)
                            print(
                                f"Saved requested frame {fnum} for camera{idx} -> {path}"
                            )
                            found = True
                            break

                    # Fallback: save oldest buffered frame
                    if not found and buffer:
                        fb_fnum, fb_frame = buffer[0]
                        path = os.path.join(
                            self.output_dir, f"camera{idx}", f"frame_{fb_fnum}.jpg"
                        )
                        cv2.imwrite(path, fb_frame)
                        print(
                            f"Warning: Frame {frame_to_save} not found in buffer for camera{idx}. "
                            f"Saved fallback frame {fb_fnum} -> {path}"
                        )

                self.frames_saved += 1

            self.frame_counter += 1
            await asyncio.sleep(0.05)

        self._cleanup()

    def stop(self) -> None:
        """Signal the streaming loop to stop."""
        self.running = False

    def _cleanup(self) -> None:
        """Release all cameras and clear all buffers."""
        for cap in self.captures.values():
            if cap.isOpened():
                cap.release()
        self.captures.clear()
        for buffer in self.frame_buffers.values():
            buffer.clear()

    # ---------------- Utilities ----------------

    def get_saved_images_base64(self) -> Dict[str, List[str]]:
        """
        Return a dict mapping camera folders ('camera0', 'camera1', ...)
        to lists of base64-encoded JPEG previews (sorted by frame number).
        """
        camera_images: Dict[str, List[str]] = {}

        if not os.path.exists(self.output_dir):
            print(f"Directory {self.output_dir} does not exist.")
            return camera_images

        def extract_frame_number(path: str) -> int:
            m = re.search(r"frame_(\d+)\.jpg", os.path.basename(path))
            return int(m.group(1)) if m else -1

        for cam_folder in os.listdir(self.output_dir):
            cam_path = os.path.join(self.output_dir, cam_folder)
            if not os.path.isdir(cam_path):
                continue

            image_paths = sorted(
                glob.glob(os.path.join(cam_path, "frame_*.jpg")), key=extract_frame_number
            )

            b64_list: List[str] = []
            for img_path in image_paths:
                try:
                    with open(img_path, "rb") as f:
                        b64_list.append(base64.b64encode(
                            f.read()).decode("utf-8"))
                except Exception as exc:
                    print(f"Failed to encode image {img_path}: {exc}")

            camera_images[cam_folder] = b64_list

        return camera_images

    # ---------------- Calibration ----------------

    def _calibrate_camera_pair(
        self,
        cam_a: int,
        cam_b: int,
        K1: np.ndarray,
        dist1: np.ndarray | List[float],
        K2: np.ndarray,
        dist2: np.ndarray | List[float],
        pattern_size: Tuple[int, int],
        square_size: float,
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Stereo-calibrate two cameras with fixed intrinsics (K, dist),
        returning (R, t) from cam_a to cam_b if successful.
        """
        try:
            folder1 = os.path.join(self.output_dir, f"camera{cam_a}")
            folder2 = os.path.join(self.output_dir, f"camera{cam_b}")

            def extract_frame_number(path: str) -> int:
                m = re.search(r"frame_(\d+)\.jpg", os.path.basename(path))
                return int(m.group(1)) if m else -1

            frames1 = sorted(glob.glob(os.path.join(
                folder1, "frame_*.jpg")), key=extract_frame_number)
            frames2 = sorted(glob.glob(os.path.join(
                folder2, "frame_*.jpg")), key=extract_frame_number)

            min_pairs = min(len(frames1), len(frames2))
            obj_points: List[np.ndarray] = []
            img_points1: List[np.ndarray] = []
            img_points2: List[np.ndarray] = []

            # Build chessboard grid (cols, rows)
            objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
            objp[:, :2] = np.mgrid[0:pattern_size[0],
                                   0:pattern_size[1]].T.reshape(-1, 2)
            objp *= float(square_size)

            for i in range(min_pairs):
                img1 = cv2.imread(frames1[i], cv2.IMREAD_GRAYSCALE)
                img2 = cv2.imread(frames2[i], cv2.IMREAD_GRAYSCALE)
                if img1 is None or img2 is None:
                    print(
                        f"Skipped pair {i}: failed to read one or both images.")
                    continue

                ret1, corners1 = cv2.findChessboardCorners(img1, pattern_size)
                ret2, corners2 = cv2.findChessboardCorners(img2, pattern_size)

                if ret1 and ret2:
                    obj_points.append(objp)
                    img_points1.append(corners1)
                    img_points2.append(corners2)
                else:
                    print(
                        f"Skipped pair {i}: checkerboard not found in both views.")

            if len(obj_points) < 5:
                print("Not enough valid image pairs.")
                return None

            image_size = img1.shape[::-1]  # (width, height)

            ret, _, _, _, _, R, t, _, _ = cv2.stereoCalibrate(
                obj_points,
                img_points1,
                img_points2,
                np.array(K1, dtype=np.float64),
                np.array(dist1, dtype=np.float64),
                np.array(K2, dtype=np.float64),
                np.array(dist2, dtype=np.float64),
                image_size,
                flags=cv2.CALIB_FIX_INTRINSIC,
            )

            print(
                f"Stereo RMS error for camera {cam_a} <-> {cam_b}: {ret:.4f}")
            return R, t

        except Exception as exc:
            print(f"Calibration error for camera pair {cam_a}, {cam_b}: {exc}")
            return None

    def calibrate_all_extrinsics(
        self,
        intrinsics: Dict[int, Dict[str, object]],
        pattern_size: Tuple[int, int] = Config.CHESSBOARD_PATTERN_SIZE,
        square_size: float = Config.CHESSBOARD_SQUARE_SIZE,
    ) -> Dict[int, Optional[Dict[str, List[List[float]]]]]:
        """
        Calibrate all cameras relative to the reference (first index by default).

        Args:
            intrinsics: Dict mapping camera index -> {"K": ..., "dist_coef"/"distCoef": ...}
                        K can be list or ndarray; dist can be list or ndarray.
        Returns:
            extrinsics: Dict[camera_index] -> {"R": [[...]*3], "t": [[...],[...],[...]]} or None on failure.
                        Reference camera gets identity R and zero t.
        """
        reference_camera = self.camera_indexes[0]
        extrinsics: Dict[int, Optional[Dict[str, List[List[float]]]]] = {}

        # Reference camera: Identity rotation and zero translation
        extrinsics[reference_camera] = {
            "R": np.eye(3).tolist(),
            "t": np.zeros((3, 1)).tolist(),
        }

        # Pull reference intrinsics (accept both 'dist_coef' and legacy 'distCoef')
        ref_K = np.array(intrinsics[reference_camera]["K"], dtype=np.float64)
        ref_dist = np.array(
            intrinsics[reference_camera].get(
                "dist_coef", intrinsics[reference_camera].get("distCoef", [])),
            dtype=np.float64,
        )

        for cam_id in self.camera_indexes:
            if cam_id == reference_camera:
                continue

            cam_K = np.array(intrinsics[cam_id]["K"], dtype=np.float64)
            cam_dist = np.array(
                intrinsics[cam_id].get(
                    "dist_coef", intrinsics[cam_id].get("distCoef", [])),
                dtype=np.float64,
            )

            print(
                f"Calibrating camera {cam_id} relative to camera {reference_camera}...")
            result = self._calibrate_camera_pair(
                reference_camera, cam_id, ref_K, ref_dist, cam_K, cam_dist, pattern_size, square_size
            )

            if result is not None:
                R, t = result
                extrinsics[cam_id] = {"R": R.tolist(), "t": t.tolist()}
            else:
                print(
                    f"Calibration failed for camera pair: {reference_camera} <-> {cam_id}")
                extrinsics[cam_id] = None

        return extrinsics

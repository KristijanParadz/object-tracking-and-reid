import os
import cv2
import base64
import shutil
import asyncio
import glob
import re
from collections import deque
from typing import Deque, Dict, Tuple
from frame_processing.config import Config


class ExtrinsicCameraStreamer:
    def __init__(self, sio, camera_indexes, output_dir='calibration/extrinsic_images'):
        self.sio = sio
        self.camera_indexes = camera_indexes
        self.output_dir = output_dir
        self.captures = {}
        self.frame_buffers: Dict[int, Deque[Tuple[int, any]]] = {
            idx: deque(maxlen=30) for idx in camera_indexes
        }
        self.frame_counter = 0
        self.frames_saved = 0
        self.running = False

    def encode_frame_to_base64(self, frame):
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            return None
        return base64.b64encode(buffer).decode('utf-8')

    async def start(self):
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)
            print(f"Cleared folder: {self.output_dir}")

        os.makedirs(self.output_dir, exist_ok=True)

        for idx in self.camera_indexes:
            cap = cv2.VideoCapture(idx)
            if not cap.isOpened():
                raise RuntimeError(f"Error: Cannot open camera index {idx}")
            self.captures[idx] = cap
            os.makedirs(os.path.join(self.output_dir,
                        f'camera{idx}'), exist_ok=True)

        self.running = True

        while self.running:
            images_base64 = {}
            current_frames = {}

            for idx, cap in self.captures.items():
                ret, frame = cap.read()
                if not ret:
                    print(
                        f"Error: Failed to grab frame from camera{idx}. Stopping.")
                    self.stop()
                    return

                self.frame_buffers[idx].append(
                    (self.frame_counter, frame.copy()))
                current_frames[idx] = frame

                encoded = self.encode_frame_to_base64(frame)
                if not encoded:
                    print(
                        f"Error: Failed to encode frame from camera{idx}. Stopping.")
                    self.stop()
                    return

                images_base64[f'camera{idx}'] = encoded

            await self.sio.emit('live-feed-extrinsic', {
                'images': images_base64,
                'frame_number': self.frame_counter,
                'frames_saved': self.frames_saved
            })

            # === Save-on-request logic ===
            if Config.EXTRINSIC_FRAME_REQUESTED:
                Config.EXTRINSIC_FRAME_REQUESTED = False
                frame_to_save = Config.EXTRINSIC_FRAME_NUMBER_TO_SAVE

                for idx, buffer in self.frame_buffers.items():
                    found = False
                    for fnum, frame in buffer:
                        if fnum == frame_to_save:
                            path = os.path.join(
                                self.output_dir, f'camera{idx}', f'frame_{fnum}.jpg')
                            cv2.imwrite(path, frame)
                            print(
                                f"Saved requested frame {fnum} for camera{idx} -> {path}")
                            found = True
                            break

                    if not found and buffer:
                        fallback_fnum, fallback_frame = buffer[0]
                        path = os.path.join(
                            self.output_dir, f'camera{idx}', f'frame_{fallback_fnum}.jpg')
                        cv2.imwrite(path, fallback_frame)
                        print(f"Warning: Frame {frame_to_save} not found in buffer for camera{idx}. "
                              f"Saved fallback frame {fallback_fnum} -> {path}")

                self.frames_saved += 1

            self.frame_counter += 1
            await asyncio.sleep(0.05)

        self._cleanup()

    def stop(self):
        self.running = False

    def _cleanup(self):
        for cap in self.captures.values():
            if cap.isOpened():
                cap.release()
        self.captures.clear()
        for buffer in self.frame_buffers.values():
            buffer.clear()

    def get_saved_images_base64(self):
        """
        Returns a dictionary where keys are camera identifiers (e.g., 'camera0')
        and values are lists of base64-encoded image previews sorted by frame number.
        """
        camera_images = {}

        if not os.path.exists(self.output_dir):
            print(f"Directory {self.output_dir} does not exist.")
            return camera_images

        def extract_frame_number(path):
            match = re.search(r'frame_(\d+)\.jpg', os.path.basename(path))
            return int(match.group(1)) if match else -1

        for cam_folder in os.listdir(self.output_dir):
            cam_path = os.path.join(self.output_dir, cam_folder)
            if not os.path.isdir(cam_path):
                continue

            image_paths = sorted(
                glob.glob(os.path.join(cam_path, 'frame_*.jpg')),
                key=extract_frame_number
            )

            base64_list = []
            for img_path in image_paths:
                try:
                    with open(img_path, 'rb') as image_file:
                        encoded = base64.b64encode(
                            image_file.read()).decode('utf-8')
                        base64_list.append(encoded)
                except Exception as e:
                    print(f"Failed to encode image {img_path}: {e}")

            camera_images[cam_folder] = base64_list

        return camera_images

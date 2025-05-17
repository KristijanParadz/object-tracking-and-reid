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

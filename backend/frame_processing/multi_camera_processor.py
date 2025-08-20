from __future__ import annotations

import asyncio
from typing import Any, Mapping, Optional

import cv2
import numpy as np
import socketio
import torch

from frame_processing.config import Config
from frame_processing.global_id_manager import GlobalIDManager
from frame_processing.single_camera_tracker import YOLOVideoTracker
from calibration.camera_calibration import CalibrationParameters
from interfaces.protocols import GlobalIDManagerProtocol
from decorators.log_call import log_call


class MultiCameraProcessor:
    """
    Coordinates multiple YOLO trackers, one per camera index, and streams
    frames to the frontend via Socket.IO.
    """

    camera_indexes: list[str]
    sio: socketio.AsyncServer
    paused: bool
    stopped: bool
    global_manager: GlobalIDManagerProtocol
    calibration_data: list[Mapping[str, Any]]
    trackers: list[YOLOVideoTracker]

    def __init__(
        self,
        camera_indexes: list[str],
        sio: socketio.AsyncServer,
        calibration_data: list[Mapping[str, Any]],
    ) -> None:
        self.camera_indexes = camera_indexes
        self.sio = sio
        self.paused = False
        self.stopped = False
        self.global_manager = GlobalIDManager(calibration_data)
        self.calibration_data = calibration_data
        self._init_trackers()

    @log_call
    def _init_trackers(self) -> None:
        """Initialize a YOLO tracker for each camera index."""
        self.trackers = []
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")

        for camera_index in self.camera_indexes:
            calib = self.find_camera_by_index(camera_index)
            if calib is None:
                # Defensive check: skip missing calibration entries gracefully
                print(f"Warning: no calibration for camera {camera_index}")
                continue

            # Accept both legacy "distCoef" and new "dist_coef" keys.
            dist = calib.get("dist_coef", calib.get("distCoef"))

            # Convert to our typed CalibrationParameters (PEP 484/526).
            params = CalibrationParameters(
                K=np.array(calib["K"], dtype=float),
                dist_coef=np.array(dist, dtype=float),
                R=np.array(calib["R"], dtype=float),
                t=np.array(calib["t"], dtype=float).reshape(3, 1),
            )

            tracker = YOLOVideoTracker(
                camera_index=camera_index,
                sio=self.sio,
                video_id=f"Camera {camera_index}",
                global_manager=self.global_manager,
                device=device,
                calibration_params=params,
            )
            self.trackers.append(tracker)

    def find_camera_by_index(self, index: str) -> Optional[Mapping[str, Any]]:
        """
        Return the raw calibration dict for a camera index, or None if absent.

        Accepts calibration entries shaped like:
          {"index": "0", "K": ..., "distCoef"/"dist_coef": ..., "R": ..., "t": ...}
        """
        for camera in self.calibration_data:
            if str(camera.get("index")) == str(index):
                return camera
        return None

    async def _process_tracker(self, tracker: YOLOVideoTracker) -> bool:
        """
        Read one frame from a tracker, process or redraw, and emit to frontend.

        Returns:
            True if a frame was processed/emitted, False if stream ended.
        """
        if not tracker.cap.isOpened():
            return False

        success, original_frame = tracker.cap.read()
        if not success:
            self._release_tracker(tracker)
            return False

        tracker.frame_counter += 1
        frame = cv2.resize(original_frame, Config.RESIZE_SHAPE)

        frame_for_frontend = None
        if (
            Config.RESIZE_SHAPE_FOR_FRONTEND != Config.RESIZE_SHAPE
            and Config.FULLSCREEN_CAMERA == tracker.video_id
        ):
            frame_for_frontend = cv2.resize(
                original_frame, Config.RESIZE_SHAPE_FOR_FRONTEND
            )

        if tracker.frame_counter % Config.SKIP_INTERVAL == 0:
            processed_frame = tracker.process_frame(
                frame, frame_for_frontend=frame_for_frontend
            )
        else:
            processed_frame = tracker.draw_last_boxes(
                frame, frame_for_frontend=frame_for_frontend
            )

        await tracker.send_image_to_frontend(processed_frame)
        return True

    def _release_tracker(self, tracker: YOLOVideoTracker) -> None:
        """Release the capture for a single tracker if open."""
        if tracker.cap.isOpened():
            tracker.cap.release()

    def _release_all_trackers(self) -> None:
        """Release all trackers' captures."""
        for tracker in self.trackers:
            self._release_tracker(tracker)

    @log_call
    async def run(self) -> None:
        """Run the main multi-camera processing loop."""
        self.stopped = False
        try:
            while not self.stopped:
                if self.paused:
                    # Light sleep to yield control when paused.
                    await asyncio.sleep(0.05)
                    continue

                any_frame_ok = False
                for tracker in self.trackers:
                    frame_processed = await self._process_tracker(tracker)
                    if frame_processed:
                        any_frame_ok = True

                # If no tracker produced a frame, exit the loop.
                if not any_frame_ok:
                    break
        finally:
            self._release_all_trackers()

    @log_call
    def stop(self) -> None:
        """Stop processing and release all resources."""
        self.stopped = True
        self._release_all_trackers()
        self.trackers = []

    @log_call
    def pause(self) -> None:
        """Pause the processing loop."""
        self.paused = True

    @log_call
    def resume(self) -> None:
        """Resume the processing loop."""
        self.paused = False

    @log_call
    async def reset(self) -> None:
        """
        Reset the processor:
        - stop current processing,
        - reset the global manager,
        - reinitialize trackers,
        - restart the processing loop in the background.
        """
        self.stop()
        await asyncio.sleep(0.1)
        self.global_manager.reset()
        self._init_trackers()
        self.stopped = False
        self.paused = False
        asyncio.create_task(self.run())

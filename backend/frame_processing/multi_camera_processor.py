import asyncio
import cv2
import torch
import socketio
from pathlib import Path
from frame_processing.global_id_manager import GlobalIDManager
from frame_processing.single_camera_tracker import YOLOVideoTracker
from frame_processing.config import Config
from calibration.camera_calibration import CameraCalibration, CalibrationParameters


class MultiCameraProcessor:
    def __init__(self, video_paths: list[str], sio: socketio.AsyncServer) -> None:
        self.video_paths = video_paths
        self.sio = sio
        self.paused = False
        self.stopped = False
        self.global_manager = GlobalIDManager()
        self._init_trackers()

    def _init_trackers(self) -> None:
        """Initializes trackers for each video path."""
        self.trackers = []
        camera_calibration = CameraCalibration("calibration/calibration.json")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")
        for video_path in self.video_paths:
            video_id = Path(video_path).stem
            calib = camera_calibration.cameras[video_id]
            tracker = YOLOVideoTracker(
                video_path=video_path,
                sio=self.sio,
                video_id=video_id,
                global_manager=self.global_manager,
                device=device,
                calibration_params=CalibrationParameters(
                    calib["K"], calib["distCoef"], calib["R"], calib["t"])
            )
            self.trackers.append(tracker)

    async def _process_tracker(self, tracker: YOLOVideoTracker) -> bool:
        """
        Processes a single tracker frame and sends it to the frontend.
        Returns True if a frame was processed, False otherwise.
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
        if Config.RESIZE_SHAPE_FOR_FRONTEND != Config.RESIZE_SHAPE and Config.FULLSCREEN_CAMERA == tracker.video_id:
            frame_for_frontend = cv2.resize(
                original_frame, Config.RESIZE_SHAPE_FOR_FRONTEND)

        if tracker.frame_counter % Config.SKIP_INTERVAL == 0:
            processed_frame = tracker.process_frame(
                frame, frame_for_frontend=frame_for_frontend)
        else:
            processed_frame = tracker.draw_last_boxes(
                frame, frame_for_frontend=frame_for_frontend)

        await tracker.send_image_to_frontend(processed_frame)
        return True

    def _release_tracker(self, tracker: YOLOVideoTracker) -> None:
        """Releases the capture of a single tracker if open."""
        if tracker.cap.isOpened():
            tracker.cap.release()

    def _release_all_trackers(self) -> None:
        """Releases all trackers' captures."""
        for tracker in self.trackers:
            self._release_tracker(tracker)

    async def run(self) -> None:
        """Runs the multi-camera processing loop."""
        self.stopped = False
        try:
            while not self.stopped:
                if self.paused:
                    await asyncio.sleep(0.05)
                    continue

                any_frame_ok = False
                for tracker in self.trackers:
                    frame_processed = await self._process_tracker(tracker)
                    if frame_processed:
                        any_frame_ok = True

                if not any_frame_ok:
                    break
        finally:
            self._release_all_trackers()

    def stop(self) -> None:
        """Stops processing and releases all resources."""
        self.stopped = True
        self._release_all_trackers()
        self.trackers = []

    def pause(self) -> None:
        """Pauses the processing loop."""
        self.paused = True

    def resume(self) -> None:
        """Resumes the processing loop."""
        self.paused = False

    async def reset(self) -> None:
        """
        Resets the processor by stopping current processing,
        resetting the global manager, reinitializing trackers,
        and starting the processing loop.
        """
        self.stop()
        await asyncio.sleep(0.1)
        self.global_manager.reset()
        self._init_trackers()
        self.stopped = False
        self.paused = False
        asyncio.create_task(self.run())

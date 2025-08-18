from __future__ import annotations

import base64
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional

import cv2
import numpy as np
import socketio
import torch
from numpy.typing import NDArray
from ultralytics import YOLO

from calibration.camera_calibration import CalibrationParameters
from frame_processing.config import Config
from frame_processing.global_id_manager import ClassID, GlobalIDManager, ObjectID
from frame_processing.re_id_model import ReIDModel
from frame_processing.utils import Point


@dataclass
class LocalTrackEntry:
    """
    Track data stored per *local* tracker ID (YOLO-assigned).
    """
    class_id: ClassID
    global_id: ObjectID
    embedding: NDArray[np.float32]
    last_update_frame: int


@dataclass
class BoundingBox:
    """Simple pixel bounding box (integer coordinates)."""
    x1: int
    y1: int
    x2: int
    y2: int


class YOLOVideoTracker:
    """
    YOLO-based object detector + Re-ID model + global tracker integration.

    - Each instance is tied to a single video/camera stream.
    - Tracks objects frame-to-frame, assigns global IDs via GlobalIDManager,
      and draws/streams annotated frames.
    """

    sio: socketio.AsyncServer
    video_id: str
    device: torch.device
    model: YOLO
    cap: cv2.VideoCapture
    frame_counter: int
    tracks: Dict[ObjectID, LocalTrackEntry]
    last_bounding_boxes: List[BoundingBox]
    last_local_ids: List[ObjectID]
    reid_model: ReIDModel
    global_manager: GlobalIDManager
    calibration_params: CalibrationParameters

    def __init__(
        self,
        camera_index: str,
        sio: socketio.AsyncServer,
        video_id: str,
        global_manager: GlobalIDManager,
        device: torch.device,
        calibration_params: CalibrationParameters,
    ) -> None:
        self.sio = sio
        self.video_id = video_id
        self.device = device
        self.model = YOLO(Config.YOLO_MODEL_PATH).to(self.device)
        self.cap = cv2.VideoCapture(camera_index)
        self.frame_counter = -1
        self.tracks = {}
        self.last_bounding_boxes = []
        self.last_local_ids = []
        self.reid_model = ReIDModel(self.device)
        self.global_manager = global_manager
        self.calibration_params = calibration_params

    # ---------------- Geometry ----------------

    def undistort_point(self, point: Point) -> Point:
        """
        Apply camera calibration to undistort a pixel coordinate.

        Args:
            point: Raw (x, y) pixel position.

        Returns:
            Point: Undistorted (x, y) position in pixel space.
        """
        pts = np.array([[[point.x, point.y]]], dtype=np.float32)
        undistorted_pts = cv2.undistortPoints(
            pts,
            self.calibration_params.K,
            # <-- updated to match PEP 526 CalibrationParameters
            self.calibration_params.dist_coef,
            P=self.calibration_params.K,
        )
        x, y = undistorted_pts[0, 0]
        return Point(x=x, y=y)

    # ---------------- Track management ----------------

    def _remove_local_duplicates(self) -> None:
        """
        Clean up duplicate local IDs pointing to the same global ID.
        Keeps the first occurrence, drops later ones.
        """
        active_ids = set(self.last_local_ids)
        self.tracks = {
            local_id: track
            for local_id, track in self.tracks.items()
            if local_id in active_ids
        }

        # Group local IDs by their assigned global ID
        global_to_locals: Dict[ObjectID, List[ObjectID]] = defaultdict(list)
        for local_id in self.last_local_ids:
            global_id = self.tracks[local_id].global_id
            global_to_locals[global_id].append(local_id)

        # Remove duplicates (preserve first)
        for local_ids in global_to_locals.values():
            for duplicate_id in local_ids[1:]:
                self.tracks.pop(duplicate_id, None)

        # Sync bounding boxes and local IDs to cleaned tracks
        updated_boxes, updated_ids = [], []
        for box, local_id in zip(self.last_bounding_boxes, self.last_local_ids):
            if local_id in self.tracks:
                updated_boxes.append(box)
                updated_ids.append(local_id)
        self.last_bounding_boxes = updated_boxes
        self.last_local_ids = updated_ids

    # ---------------- Frame processing ----------------

    def process_frame(
        self, frame: NDArray[np.uint8], frame_for_frontend: Optional[NDArray[np.uint8]] = None
    ) -> NDArray[np.uint8]:
        """
        Detect objects, update tracks, and overlay bounding boxes.

        Args:
            frame: Current video frame.
            frame_for_frontend: Optional resized frame for display (annotations scaled).

        Returns:
            Frame with drawn bounding boxes and IDs.
        """
        self.last_bounding_boxes.clear()
        self.last_local_ids.clear()

        results = self.model.track(
            frame, persist=True, device=self.device, verbose=False)
        if not results or not results[0].boxes:
            return self.draw_last_boxes(frame)

        boxes = results[0].boxes.xyxy
        ids = results[0].boxes.id
        classes = results[0].boxes.cls

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box[:4])
            local_id = int(ids[i]) if ids is not None else i
            class_id = int(classes[i]) if classes is not None else 0

            self.last_bounding_boxes.append(BoundingBox(x1, y1, x2, y2))
            self.last_local_ids.append(local_id)

            crop = frame[y1:y2, x1:x2]
            bbox_center = Point(
                (x1 * Config.SCALE + x2 * Config.SCALE) / 2,
                (y1 * Config.SCALE + y2 * Config.SCALE) / 2,
            )
            undistorted_center = self.undistort_point(bbox_center)

            if local_id not in self.tracks:
                embedding = self.reid_model.get_embedding(crop)
                global_id = self.global_manager.match_or_create(
                    embedding, class_id, self.video_id, undistorted_center, self.frame_counter
                )
                self.tracks[local_id] = LocalTrackEntry(
                    class_id=class_id,
                    global_id=global_id,
                    embedding=embedding,
                    last_update_frame=self.frame_counter,
                )
                continue

            # Existing track: update position and maybe embedding
            track_data = self.tracks[local_id]
            self.global_manager.update_position(
                track_data.global_id, self.video_id, undistorted_center, self.frame_counter
            )
            if self.frame_counter - track_data.last_update_frame < Config.EMBEDDING_UPDATE_INTERVAL:
                continue

            new_embedding = self.reid_model.get_embedding(crop)
            self.global_manager.update_embedding(
                track_data.global_id, new_embedding)
            track_data.embedding = new_embedding
            track_data.last_update_frame = self.frame_counter

        self._remove_local_duplicates()
        return self.draw_last_boxes(frame, frame_for_frontend)

    def draw_last_boxes(
        self, frame: NDArray[np.uint8], frame_for_frontend: Optional[NDArray[np.uint8]] = None
    ) -> NDArray[np.uint8]:
        """
        Draw bounding boxes + labels (global ID, optional triangulated XYZ) on the frame.

        Args:
            frame: Current video frame.
            frame_for_frontend: If provided, draw on this resized version.

        Returns:
            Annotated frame.
        """
        scale = 1.0
        if frame_for_frontend is not None:
            scale = frame_for_frontend.shape[0] / frame.shape[0]
            frame = frame_for_frontend

        for bbox, local_id in zip(self.last_bounding_boxes, self.last_local_ids):
            if local_id not in self.tracks:
                continue
            track_data = self.tracks[local_id]
            global_id = track_data.global_id
            color = self.global_manager.get_color(global_id)
            tri_pt = self.global_manager.get_triangulated_point(global_id)

            label = f"ID:{global_id}"
            if tri_pt is not None:
                x, y, z = tri_pt
                label += f" X:{round(x)} Y:{round(y)} Z:{round(z)}"

            top_left = (int(bbox.x1 * scale), int(bbox.y1 * scale))
            bottom_right = (int(bbox.x2 * scale), int(bbox.y2 * scale))
            cv2.rectangle(frame, top_left, bottom_right, color, 2)

            text_org_above = (top_left[0], top_left[1] - int(5 * scale))
            text_org = text_org_above if text_org_above[1] >= 0 else (
                bottom_right[0] - int(45 * scale),
                bottom_right[1] - int(5 * scale),
            )
            cv2.putText(frame, label, text_org,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        return frame

    # ---------------- Frontend ----------------

    async def send_image_to_frontend(self, image: NDArray[np.uint8]) -> None:
        """
        Encode an image as base64 JPEG and emit it to the frontend.
        """
        _, buffer = cv2.imencode(".jpg", image)
        base64_jpg = base64.b64encode(buffer).decode("utf-8")
        await self.sio.emit(
            "processed-images", {"video_id": self.video_id,
                                 "image": base64_jpg}
        )

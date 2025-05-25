import base64
import cv2
import numpy as np
import socketio
import torch
from collections import defaultdict
from ultralytics import YOLO
from typing import List, Dict
from frame_processing.config import Config
from frame_processing.re_id_model import ReIDModel
from frame_processing.global_id_manager import GlobalIDManager, ObjectID, ClassID
from dataclasses import dataclass
from calibration.camera_calibration import CalibrationParameters
from frame_processing.utils import Point


@dataclass
class LocalTrackEntry:
    class_id: ClassID
    global_id: ObjectID
    embedding: np.ndarray
    last_update_frame: int


@dataclass
class BoundingBox:
    x1: int
    y1: int
    x2: int
    y2: int


class YOLOVideoTracker:
    """
    A tracker that uses YOLO for object detection and a re-identification model for tracking objects in a video.
    """

    def __init__(self, camera_index: str, sio: socketio.AsyncServer, video_id: str, global_manager: GlobalIDManager, device: torch.device, calibration_params: CalibrationParameters):
        """
        Initializes the YOLOVideoTracker.

        Args:
            camera_index (str): Path to the input video.
            sio(socketio.AsyncServer): SocketIO instance for emitting events.
            video_id (str): Identifier for the video.
            global_manager(GlobalIDManager): Manager for global tracking across frames.
            device(torch.device): Device to run computations on (e.g. 'cpu' or 'cuda').
        """
        self.sio = sio
        self.video_id = video_id
        self.device = device
        self.model = YOLO(Config.YOLO_MODEL_PATH).to(self.device)
        self.cap = cv2.VideoCapture(camera_index)
        self.frame_counter = -1
        self.tracks: Dict[ObjectID, LocalTrackEntry] = {}
        self.last_bounding_boxes: List[BoundingBox] = []
        self.last_local_ids: List[ObjectID] = []
        self.reid_model = ReIDModel(self.device)
        self.global_manager = global_manager
        self.calibration_params = calibration_params

    def undistort_point(self, point: Point) -> Point:
        pts = np.array([[[point.x, point.y]]], dtype=np.float32)

        undistorted_pts = cv2.undistortPoints(
            pts, self.calibration_params.K, self.calibration_params.distCoef, P=self.calibration_params.K)

        x, y = undistorted_pts[0, 0]
        return Point(x=x, y=y)

    def _remove_local_duplicates(self) -> None:
        """
        Cleans up the tracks dictionary by removing any duplicate local IDs based on their global ID
        and synchronizes the list of last bounding boxes and local IDs.
        """
        # Only keep tracks corresponding to currently active local IDs.
        active_ids = set(self.last_local_ids)
        self.tracks = {local_id: track for local_id,
                       track in self.tracks.items() if local_id in active_ids}

        # Group local IDs by their global ID.
        global_to_locals = defaultdict(list)
        for local_id in self.last_local_ids:
            global_id = self.tracks[local_id].global_id
            global_to_locals[global_id].append(local_id)

        # Remove duplicates for each global ID (preserve the first occurrence).
        for local_ids in global_to_locals.values():
            if len(local_ids) > 1:
                for duplicate_id in local_ids[1:]:
                    self.tracks.pop(duplicate_id, None)

        # Update the lists for the last bounding boxes and local IDs.
        updated_boxes = []
        updated_ids = []
        for box, local_id in zip(self.last_bounding_boxes, self.last_local_ids):
            if local_id in self.tracks:
                updated_boxes.append(box)
                updated_ids.append(local_id)
        self.last_bounding_boxes = updated_boxes
        self.last_local_ids = updated_ids

    def process_frame(self, frame: np.ndarray, frame_for_frontend: np.ndarray | None = None):
        """
        Processes a single video frame by detecting objects, updating tracks, and drawing bounding boxes.

        Args:
            frame: The current video frame (numpy array).
            frame_for_frontend: An optional frame used to scale drawing for display.

        Returns:
            The video frame with drawn bounding boxes.
        """
        # Clear detections from the previous frame.
        self.last_bounding_boxes.clear()
        self.last_local_ids.clear()

        results = self.model.track(
            frame, persist=True, device=self.device, verbose=False)

        # Early exit if no results or boxes are found.
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

            bbox_center = Point((x1 * Config.SCALE+x2 * Config.SCALE)/2,
                                (y1 * Config.SCALE+y2 * Config.SCALE)/2)
            undistorted_bbox_center = self.undistort_point(bbox_center)

            # If track doesn't exist, create a new track and skip further processing.
            if local_id not in self.tracks:
                embedding = self.reid_model.get_embedding(crop)
                global_id = self.global_manager.match_or_create(
                    embedding, class_id, self.video_id, undistorted_bbox_center, self.frame_counter)
                self.tracks[local_id] = LocalTrackEntry(
                    class_id,
                    global_id,
                    embedding,
                    self.frame_counter
                )
                continue

            track_data = self.tracks[local_id]
            self.global_manager.update_position(
                track_data.global_id, self.video_id, undistorted_bbox_center, self.frame_counter)
            # Skip updating the track if it's not time for an embedding update.
            if self.frame_counter - track_data.last_update_frame < Config.EMBEDDING_UPDATE_INTERVAL:
                continue

            new_embedding = self.reid_model.get_embedding(crop)
            self.global_manager.update_embedding(
                track_data.global_id, new_embedding)
            track_data.embedding = new_embedding
            track_data.last_update_frame = self.frame_counter

        self._remove_local_duplicates()
        return self.draw_last_boxes(frame, frame_for_frontend)

    def draw_last_boxes(self, frame: np.ndarray, frame_for_frontend: np.ndarray | None = None):
        """
        Draws bounding boxes and global IDs on the frame for the latest detections.
        All drawn elements are scaled by the provided scale factor.

        Args:
            frame: The video frame (numpy array).
            frame_for_frontend: An optional frame used to scale drawing for display.

        Returns:
            The video frame with drawn bounding boxes and IDs.
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
            triangulated_point = self.global_manager.get_triangulated_point(
                global_id)

            label = f"ID:{global_id}"
            if triangulated_point is not None:
                [x, y, z] = triangulated_point
                label = f"{label} X:{round(x)} Y:{round(y)} Z:{round(z)}"

            top_left = (int(bbox.x1 * scale), int(bbox.y1 * scale))
            bottom_right = (int(bbox.x2 * scale), int(bbox.y2 * scale))

            cv2.rectangle(frame, top_left, bottom_right, color, 2)

            text_org_above = (top_left[0],
                              top_left[1] - int(5 * scale))

            # If placing text above goes off screen, put it at the bottom-right
            if text_org_above[1] < 0:
                text_org = (bottom_right[0] - int(45 * scale),
                            bottom_right[1] - int(5 * scale))
            else:
                text_org = text_org_above

            cv2.putText(frame, label, text_org,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        return frame

    async def send_image_to_frontend(self, image: np.ndarray) -> None:
        """
        Encodes an image to JPEG, converts it to a base64 string, and sends it to the frontend.

        Args:
            image: The image (numpy array) to be sent.
        """
        _, buffer = cv2.imencode('.jpg', image)
        base64_jpg = base64.b64encode(buffer).decode('utf-8')
        await self.sio.emit('processed-images', {"video_id": self.video_id, "image": base64_jpg})

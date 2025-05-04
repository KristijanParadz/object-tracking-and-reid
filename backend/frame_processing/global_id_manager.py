import numpy as np
import random
from typing import Dict, Tuple, Optional, List, Any
from dataclasses import dataclass
from frame_processing.config import Config
from frame_processing.utils import Point
from calibration.camera_calibration import CameraCalibration

Color = Tuple[int, int, int]
ClassID = int
ObjectID = int
CameraId = str
CalibrationData = Dict[str, Any]  # K, distCoef, R, t


@dataclass
class Position:
    point: Point
    last_frame_updated: int


@dataclass
class GlobalTrackEntry:
    """
    GlobalTrackEntry holds an embedding, color, and observed positions.
    """
    embedding: np.ndarray
    color: Color
    # Stores the *last known* position in each camera
    positions: Dict[CameraId, Position]
    triangulated_point: np.ndarray | None = None


class GlobalIDManager:
    """
    GlobalIDManager manages global IDs using embeddings and spatial constraints.

    Attributes:
        global_tracks (Dict[ClassID, Dict[ObjectID, GlobalTrackEntry]]):
            Mapping of class IDs to global IDs and their track data.
        global_id_to_class (Dict[ObjectID, ClassID]): Mapping from global IDs to class IDs.
        next_global_id (ObjectID): Counter for the next available global ID.
        calibration_data (Dict[CameraId, CalibrationData]): Camera calibration info.
        fundamental_matrices (Dict[Tuple[CameraId, CameraId], np.ndarray]): Cache for F matrices.
    """

    def __init__(self) -> None:
        """
        Initializes the GlobalIDManager.

        Args:
            calibration_data (Dict[CameraId, CalibrationData]):
                A dictionary mapping camera IDs to their calibration data (K, distCoef, R, t).
                'K', 'R', 't' should be numpy arrays. 'distCoef' can be list or array.
        """
        self.global_tracks: Dict[ClassID,
                                 Dict[ObjectID, GlobalTrackEntry]] = {}
        self.global_id_to_class: Dict[ObjectID, ClassID] = {}
        self.next_global_id: ObjectID = 1
        self.calibration_data = self._validate_calibration(
            CameraCalibration("calibration/calibration.json").cameras)
        self.fundamental_matrices: Dict[Tuple[CameraId,
                                              CameraId], Optional[np.ndarray]] = {}

    def compute_epipolar_line(self, F: np.ndarray, point: Point):
        point_h = np.array([point.x, point.y, 1]
                           )  # Convert to homogeneous coordinates
        line = F @ point_h  # Compute epipolar line equation
        return line / np.linalg.norm(line[:2])  # Normalize line

    def distance_point_to_line(self, point: Point, line: np.ndarray):
        return abs(line[0] * point.x + line[1] * point.y + line[2])

    def _validate_calibration(self, calib_data: Dict[CameraId, CalibrationData]) -> Dict[CameraId, CalibrationData]:
        """Ensures calibration data has numpy arrays."""
        validated_data = {}
        for cam_id, data in calib_data.items():
            try:
                validated_data[cam_id] = {
                    "K": np.array(data["K"], dtype=float),
                    "distCoef": np.array(data["distCoef"], dtype=float),
                    "R": np.array(data["R"], dtype=float),
                    # Ensure t is 3x1
                    "t": np.array(data["t"], dtype=float).reshape(3, 1)
                }
                # Basic shape checks
                assert validated_data[cam_id]["K"].shape == (3, 3)
                assert validated_data[cam_id]["R"].shape == (3, 3)
                assert validated_data[cam_id]["t"].shape == (3, 1)
            except (KeyError, TypeError, AssertionError) as e:
                raise ValueError(
                    f"Invalid calibration data format for camera {cam_id}: {e}")
        return validated_data

    def _get_fundamental_matrix(self, cam1_id: CameraId, cam2_id: CameraId) -> Optional[np.ndarray]:
        """
        Computes or retrieves the Fundamental Matrix F from cam1 to cam2.
        Points in cam1 image are mapped to epipolar lines in cam2 image by l2 = F @ p1.
        Returns None if calibration data is missing.
        """
        if cam1_id == cam2_id:
            return None  # No fundamental matrix for the same camera
        if (cam1_id, cam2_id) in self.fundamental_matrices:
            return self.fundamental_matrices[(cam1_id, cam2_id)]
        if cam1_id not in self.calibration_data or cam2_id not in self.calibration_data:
            return None

        calib1 = self.calibration_data[cam1_id]
        calib2 = self.calibration_data[cam2_id]

        K1, R1, t1 = calib1["K"], calib1["R"], calib1["t"].flatten()
        K2, R2, t2 = calib2["K"], calib2["R"], calib2["t"].flatten()

        # Calculate relative pose (from cam1 to cam2)
        # R: Rotation from cam1 coords to cam2 coords
        # t: Translation from cam1 origin to cam2 origin, expressed in cam2 coords
        try:
            R_rel = R2 @ R1.T
            t_rel = t2 - R2 @ R1.T @ t1

            # Skew-symmetric matrix of t_rel
            t_x = np.array([
                [0, -t_rel[2], t_rel[1]],
                [t_rel[2], 0, -t_rel[0]],
                [-t_rel[1], t_rel[0], 0]
            ])

            # Essential matrix
            E = t_x @ R_rel

            # Fundamental matrix
            F = np.linalg.inv(K2).T @ E @ np.linalg.inv(K1)

            self.fundamental_matrices[(cam1_id, cam2_id)] = F
            # Also cache the inverse relation
            self.fundamental_matrices[(cam2_id, cam1_id)] = F.T
            return F

        except np.linalg.LinAlgError as e:
            print(
                f"Error calculating fundamental matrix between {cam1_id} and {cam2_id}: {e}")
            self.fundamental_matrices[(cam1_id, cam2_id)] = None
            self.fundamental_matrices[(cam2_id, cam1_id)] = None
            return None

    def _calculate_distance(self, pt1: Point, pt2: Point, F: np.ndarray) -> float:
        line = self.compute_epipolar_line(F, pt1)
        dist = self.distance_point_to_line(pt2, line)
        return dist

    def _is_spatially_consistent(self, current_cam: CameraId, current_pt: Point, track_positions: Dict[CameraId, Position], frame_number) -> bool:
        """
        Checks if the current point is spatially consistent with any previous observations
        of the track in *other* cameras using epipolar geometry.
        """
        if not track_positions:  # No previous positions to compare with
            return True  # Cannot perform check, assume consistent for now

        is_consistent = False
        # Check against positions from *other* cameras
        other_camera_positions = {
            cam: pos for cam, pos in track_positions.items() if cam != current_cam and frame_number - pos.last_frame_updated <= 10}

        if not other_camera_positions:
            # Only seen in the current camera before (or first time seen)
            # Spatial check N/A for matching, rely on embedding.
            # Or if the track exists but only in the current cam, the check is not applicable for matching purpose across cameras.
            return True  # Cannot perform cross-camera check

        for prev_cam_id, prev_pt in other_camera_positions.items():
            # Check consistency: current detection vs previous detection in another camera
            # F maps points from prev_cam to lines in current_cam
            F = self._get_fundamental_matrix(prev_cam_id, current_cam)
            if F is not None:
                # Ensure points are in the correct format (assuming Point has x, y attributes)
                distance = self._calculate_distance(
                    prev_pt.point, current_pt, F)

                if distance < Config.EPIPOLAR_THRESHOLD:  # Compare distance squared
                    is_consistent = True
                    break  # Found one consistent view, that's enough

        # print(f"  Overall Spatial Consistency for this track: {is_consistent}")
        return is_consistent

    def _generate_random_color(self) -> Color:
        """Generates a random color represented as an RGB tuple."""
        return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

    def match_or_create(self, embedding: np.ndarray, class_id: ClassID, camera_id: CameraId, bbox_center: Point, frame_number: int) -> ObjectID:
        """
        Matches an embedding to an existing global ID using similarity and spatial constraints,
        or creates a new one if no suitable match is found.

        Args:
            embedding (np.ndarray): The input embedding to match.
            class_id (ClassID): The identifier for the class/category.
            camera_id (CameraId): The ID of the camera observing this detection.
            bbox_center (Point): The undistorted center point of the bounding box.

        Returns:
            ObjectID: The matched or newly created global ID.
        """
        if not isinstance(embedding, np.ndarray):
            raise TypeError("embedding must be a numpy array.")
        if camera_id not in self.calibration_data:
            raise ValueError(
                f"Calibration data missing for camera {camera_id}")

        if class_id not in self.global_tracks:
            self.global_tracks[class_id] = {}

        potential_matches: List[Tuple[float, ObjectID]] = []

        # 1. Find all candidates above similarity threshold
        for g_id, track in self.global_tracks[class_id].items():
            sim: float = float(np.dot(embedding, track.embedding))
            if sim >= Config.SIMILARITY_THRESHOLD:
                potential_matches.append((sim, g_id))

        # 2. Sort candidates by similarity (highest first)
        potential_matches.sort(key=lambda x: x[0], reverse=True)

        # 3. Check spatial consistency for the best candidates
        matched_g_id: Optional[ObjectID] = None
        for sim, g_id in potential_matches:
            track = self.global_tracks[class_id][g_id]
            # print(f"Checking GID {g_id} (Sim: {sim:.3f}) for spatial consistency...")
            if self._is_spatially_consistent(camera_id, bbox_center, track.positions, frame_number):
                # print(f"  --> Spatially Consistent! Matched GID: {g_id}")
                matched_g_id = g_id
                break  # Found the best match that is also spatially consistent
            # else:
                # print(f"  --> Spatially INconsistent.")

        # 4. Assign ID or create new one
        if matched_g_id is not None:
            # Update position for the matched track in the current camera
            self.global_tracks[class_id][matched_g_id].positions[camera_id] = Position(
                bbox_center, frame_number)
            # Optionally update embedding (if needed)
            self.update_embedding(matched_g_id, embedding)
            return matched_g_id
        else:
            # No suitable match found (either low similarity or spatially inconsistent)
            color: Color = self._generate_random_color()
            new_g_id: ObjectID = self.next_global_id
            self.next_global_id += 1

            # print(f"Creating new GID {new_g_id} for class {class_id} in camera {camera_id}")
            self.global_tracks[class_id][new_g_id] = GlobalTrackEntry(
                # Initialize positions dict
                embedding, color, {camera_id: Position(
                    bbox_center, frame_number)}
            )
            self.global_id_to_class[new_g_id] = class_id
            return new_g_id

    def get_color(self, global_id: ObjectID) -> Color:
        """Retrieves the color associated with a given global ID."""
        if global_id not in self.global_id_to_class:
            return (255, 255, 255)  # Default color if not found

        cls_id: ClassID = self.global_id_to_class[global_id]
        # Check existence carefully
        if cls_id in self.global_tracks and global_id in self.global_tracks[cls_id]:
            return self.global_tracks[cls_id][global_id].color
        else:
            # This case might indicate an inconsistency if global_id_to_class has the ID
            # but global_tracks doesn't. Log a warning?
            print(
                f"Warning: GID {global_id} found in global_id_to_class but not in global_tracks.")
            return (255, 0, 0)  # Return a distinct color like red for errors

    def get_triangulated_point(self, global_id: ObjectID) -> np.ndarray | None:
        """Retrieves the color associated with a given global ID."""
        if global_id not in self.global_id_to_class:
            return None  # Default color if not found

        cls_id: ClassID = self.global_id_to_class[global_id]
        # Check existence carefully
        if cls_id in self.global_tracks and global_id in self.global_tracks[cls_id]:
            return self.global_tracks[cls_id][global_id].triangulated_point
        else:
            # This case might indicate an inconsistency if global_id_to_class has the ID
            # but global_tracks doesn't. Log a warning?
            print(
                f"Warning: GID {global_id} found in global_id_to_class but not in global_tracks.")
            return None  # Return a distinct color like red for errors

    def update_embedding(self, global_id: ObjectID, new_embedding: np.ndarray, alpha: Optional[float] = None) -> None:
        """
        Updates the embedding for a given global ID by blending.
        Uses Config.EMBEDDING_ALPHA if alpha is None.
        """
        if alpha is None:
            alpha = Config.EMBEDDING_ALPHA
        if not isinstance(new_embedding, np.ndarray):
            raise TypeError("new_embedding must be a numpy array.")

        if global_id not in self.global_id_to_class:
            return

        cls_id: ClassID = self.global_id_to_class[global_id]
        if cls_id not in self.global_tracks or global_id not in self.global_tracks[cls_id]:
            return

        track = self.global_tracks[cls_id][global_id]
        blended = alpha * track.embedding + (1.0 - alpha) * new_embedding
        norm = np.linalg.norm(blended)
        if norm > 0:
            blended /= norm  # Re-normalize
        self.global_tracks[cls_id][global_id].embedding = blended

    def triangulate_multiview_svd(self, positions: Dict[CameraId, Position], current_frame_number: int) -> np.ndarray:
        """
        Triangulate a 3D point from multiple camera views using the DLT method with SVD.

        Args:
            positions: Dict[CameraId, Point] - *Undistorted* 2D points from each camera.
                                            It's crucial these points are already corrected
                                            for lens distortion.

        Returns:
            point_3d: (3,) numpy array representing the triangulated [X, Y, Z] world coordinates.
                    Returns None if triangulation fails or fewer than 2 views are available.
        """
        A = []  # Stores the rows for the linear system Ax = 0

        calibrations = self.calibration_data
        valid_views = 0
        for cam_id, position in positions.items():
            if current_frame_number - position.last_frame_updated > 10:
                continue
            calib = calibrations[cam_id]
            K = calib["K"]
            R = calib["R"]
            t = calib["t"]

            # Construct projection matrix P = K [R | t]
            P = K @ np.hstack([R, t])

            # Get undistorted pixel coordinates
            u, v = position.point.x, position.point.y

            # Extract rows of P
            p1T = P[0, :]
            p2T = P[1, :]
            p3T = P[2, :]

            # Form the two linear equations for this view:
            # (u * p3^T - p1^T) * X = 0
            # (v * p3^T - p2^T) * X = 0
            # where X = [Xw, Yw, Zw, 1]^T is the homogeneous world point
            row1 = u * p3T - p1T
            row2 = v * p3T - p2T
            A.append(row1)
            A.append(row2)
            valid_views += 1

        if valid_views < 2:
            print(
                f"Error: At least 2 valid camera views with calibrations are needed for triangulation. Found {valid_views}.")
            # Or raise ValueError("At least 2 cameras are needed for triangulation.")
            return None

        # Stack the rows into the matrix A
        A = np.vstack(A)  # Shape will be (2 * num_views, 4)

        # Solve the homogeneous system Ax = 0 using SVD
        # A = U * S * Vh (where Vh = V^T)
        # The solution x is the last column of V, which is the last row of Vh
        U, S, Vh = np.linalg.svd(A)
        point_3d_homogeneous = Vh[-1, :]  # Last row of Vh (V transpose)

        # Convert from homogeneous to Cartesian coordinates
        if point_3d_homogeneous[3] == 0:
            print(
                "Warning: Triangulation resulted in point at infinity (homogeneous W=0).")
            return None  # Or handle as appropriate

        point_3d = point_3d_homogeneous[:3] / point_3d_homogeneous[3]

        return point_3d

    # --- Add a method to update position explicitly if needed outside matching ---
    def update_position(self, global_id: ObjectID, camera_id: CameraId, bbox_center: Point, frame_number: int) -> None:
        """ Explicitly updates the last known position for a global ID in a specific camera. """
        if global_id not in self.global_id_to_class:
            return

        cls_id: ClassID = self.global_id_to_class[global_id]
        if cls_id not in self.global_tracks or global_id not in self.global_tracks[cls_id]:
            return

        self.global_tracks[cls_id][global_id].positions[camera_id] = Position(
            bbox_center, frame_number)
        triangulated_point = self.triangulate_multiview_svd(
            self.global_tracks[cls_id][global_id].positions, frame_number)
        self.global_tracks[cls_id][global_id].triangulated_point = triangulated_point

    def reset(self) -> None:
        """Resets the global tracking state."""
        self.global_tracks.clear()
        self.global_id_to_class.clear()
        self.fundamental_matrices.clear()  # Clear cache
        self.next_global_id = 1

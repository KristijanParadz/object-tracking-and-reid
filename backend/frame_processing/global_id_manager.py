from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from calibration.camera_calibration import CalibrationParameters, CameraCalibration
from frame_processing.config import Config
from frame_processing.utils import Point

# ---------- Type aliases ----------
Color = Tuple[int, int, int]
ClassID = int
ObjectID = int
CameraId = str
FloatArray = NDArray[np.float64]


@dataclass
class Position:
    """
    Last known 2D position of a track in a given camera, and the frame number
    when it was last updated.
    """
    point: Point
    last_frame_updated: int


@dataclass
class GlobalTrackEntry:
    """
    Aggregated info for a global track:
      - running (normalized) embedding
      - display color
      - per-camera last seen positions
      - optional triangulated 3D point
    """
    embedding: FloatArray
    color: Color
    positions: Dict[CameraId, Position]
    triangulated_point: Optional[FloatArray] = None


class GlobalIDManager:
    """
    Manage global IDs across multiple cameras using embedding similarity
    and epipolar-geometry consistency checks.
    """

    global_tracks: Dict[ClassID, Dict[ObjectID, GlobalTrackEntry]]
    global_id_to_class: Dict[ObjectID, ClassID]
    next_global_id: ObjectID
    # Calibrations keyed by camera name/id (e.g., "Camera 0")
    calibration_data: Dict[CameraId, CalibrationParameters]
    fundamental_matrices: Dict[Tuple[CameraId, CameraId], Optional[FloatArray]]

    def __init__(self, calibration_data: List[Dict[str, Any]]) -> None:
        """
        Args:
            calibration_data: Iterable of per-camera dicts with keys:
              "index", "K", "R", "t" and distortion as "distCoef" or "dist_coef".
              Values are converted to float arrays by CameraCalibration.
        """
        self.global_tracks = {}
        self.global_id_to_class = {}
        self.next_global_id = 1

        # Normalize + validate with our CameraCalibration loader.
        loader = CameraCalibration(calibration_data)
        # Use the parsed/typed dict directly (already validated/typed).
        self.calibration_data = loader.cameras

        # Cache for F matrices between camera pairs.
        self.fundamental_matrices = {}

    # ---------------- Epipolar geometry helpers ----------------

    @staticmethod
    def compute_epipolar_line(F: FloatArray, point: Point) -> FloatArray:
        """
        Given a fundamental matrix F (cam1 -> cam2) and a pixel point in cam1,
        compute the epipolar line l in cam2 (homogeneous 3-vector).
        """
        point_h = np.array([point.x, point.y, 1.0], dtype=float)
        line = F @ point_h
        # Normalize by line direction magnitude (a, b)
        ab_norm = np.linalg.norm(line[:2])
        return line / ab_norm if ab_norm > 0 else line

    @staticmethod
    def distance_point_to_line(point: Point, line: FloatArray) -> float:
        """Signed distance |ax + by + c| for homogeneous line [a, b, c]."""
        return float(abs(line[0] * point.x + line[1] * point.y + line[2]))

    def _get_fundamental_matrix(
        self, cam1_id: CameraId, cam2_id: CameraId
    ) -> Optional[FloatArray]:
        """
        Compute or retrieve the fundamental matrix F mapping cam1 pixels to
        epipolar lines in cam2: l2 = F @ p1.
        Returns None if same camera or calibration is missing.
        """
        if cam1_id == cam2_id:
            return None

        if (cam1_id, cam2_id) in self.fundamental_matrices:
            return self.fundamental_matrices[(cam1_id, cam2_id)]

        if cam1_id not in self.calibration_data or cam2_id not in self.calibration_data:
            return None

        calib1 = self.calibration_data[cam1_id]
        calib2 = self.calibration_data[cam2_id]

        K1, R1, t1 = calib1.K, calib1.R, calib1.t.flatten()
        K2, R2, t2 = calib2.K, calib2.R, calib2.t.flatten()

        try:
            # Relative pose from cam1 to cam2
            R_rel = R2 @ R1.T
            t_rel = t2 - R2 @ R1.T @ t1

            # Skew-symmetric matrix [t]_x
            t_x = np.array(
                [[0.0, -t_rel[2], t_rel[1]],
                 [t_rel[2], 0.0, -t_rel[0]],
                 [-t_rel[1], t_rel[0], 0.0]],
                dtype=float,
            )

            # Essential E = [t]_x R
            E = t_x @ R_rel

            # Fundamental F = K2^{-T} E K1^{-1}
            F = np.linalg.inv(K2).T @ E @ np.linalg.inv(K1)

            # Cache both directions (inverse relation is F^T)
            self.fundamental_matrices[(cam1_id, cam2_id)] = F
            self.fundamental_matrices[(cam2_id, cam1_id)] = F.T
            return F
        except np.linalg.LinAlgError as exc:
            print(
                f"Error calculating F between {cam1_id} and {cam2_id}: {exc}")
            self.fundamental_matrices[(cam1_id, cam2_id)] = None
            self.fundamental_matrices[(cam2_id, cam1_id)] = None
            return None

    def _calculate_distance(self, pt1: Point, pt2: Point, F: FloatArray) -> float:
        """Epipolar distance of pt2 in cam2 to the line from pt1 in cam1."""
        line = self.compute_epipolar_line(F, pt1)
        return self.distance_point_to_line(pt2, line)

    def _calculate_lowest_distance(
        self,
        current_cam: CameraId,
        current_pt: Point,
        track_positions: Dict[CameraId, Position],
        frame_number: int,
    ) -> Optional[float]:
        """
        Compare the current point with recent observations of this track in
        *other* cameras and return the minimum epipolar distance found
        (if under threshold). Otherwise return None.
        """
        if not track_positions:
            return None

        # Only consider other-camera positions updated in the last N frames
        other_positions = {
            cam: pos
            for cam, pos in track_positions.items()
            if cam != current_cam and frame_number - pos.last_frame_updated <= 10
        }
        if not other_positions:
            return None

        best: Optional[float] = None
        for prev_cam_id, prev_pos in other_positions.items():
            F = self._get_fundamental_matrix(prev_cam_id, current_cam)
            if F is None:
                continue
            distance = self._calculate_distance(prev_pos.point, current_pt, F)
            if distance < Config.EPIPOLAR_THRESHOLD:
                best = distance if best is None else min(best, distance)
        return best

    # ---------------- Public API ----------------

    @staticmethod
    def _random_color() -> Color:
        """Generate a random RGB color."""
        return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

    def match_or_create(
        self,
        embedding: FloatArray,
        class_id: ClassID,
        camera_id: CameraId,
        bbox_center: Point,
        frame_number: int,
    ) -> ObjectID:
        """
        Match an embedding to an existing global ID (by similarity + geometry),
        or create a new one.

        Notes:
            - `bbox_center` must be an *undistorted* pixel coordinate.
            - Requires calibration for the given camera.
        """
        if not isinstance(embedding, np.ndarray):
            raise TypeError("embedding must be a numpy array.")
        if camera_id not in self.calibration_data:
            raise ValueError(
                f"Calibration data missing for camera {camera_id}")

        if class_id not in self.global_tracks:
            self.global_tracks[class_id] = {}

        # 1) Collect candidates above the cosine-similarity threshold
        candidates: List[Tuple[float, ObjectID]] = []
        for gid, track in self.global_tracks[class_id].items():
            sim = float(np.dot(embedding, track.embedding))
            if sim >= Config.SIMILARITY_THRESHOLD:
                candidates.append((sim, gid))

        # 2) Sort by similarity (descending)
        candidates.sort(key=lambda s: s[0], reverse=True)

        # 3) Among candidates, pick the one with the best epipolar consistency
        best_match: Optional[Tuple[float, float, ObjectID]
                             ] = None  # (distance, sim, id)
        for sim, gid in candidates:
            track = self.global_tracks[class_id][gid]
            dist = self._calculate_lowest_distance(
                camera_id, bbox_center, track.positions, frame_number
            )
            if dist is not None:
                if best_match is None or dist < best_match[0]:
                    best_match = (dist, sim, gid)

        # 4) Assign or create
        if best_match is not None:
            _, _, matched_gid = best_match
            self.global_tracks[class_id][matched_gid].positions[camera_id] = Position(
                bbox_center, frame_number
            )
            self.update_embedding(matched_gid, embedding)
            return matched_gid

        # No suitable match: create a new global ID
        color = self._random_color()
        new_gid: ObjectID = self.next_global_id
        self.next_global_id += 1

        self.global_tracks[class_id][new_gid] = GlobalTrackEntry(
            embedding=embedding,
            color=color,
            positions={camera_id: Position(bbox_center, frame_number)},
        )
        self.global_id_to_class[new_gid] = class_id
        return new_gid

    def get_color(self, global_id: ObjectID) -> Color:
        """Return the RGB color for a global ID (fallbacks to white/red)."""
        if global_id not in self.global_id_to_class:
            return (255, 255, 255)  # White if not tracked

        cls_id = self.global_id_to_class[global_id]
        if cls_id in self.global_tracks and global_id in self.global_tracks[cls_id]:
            return self.global_tracks[cls_id][global_id].color

        # Inconsistency: ID mapped to class but not present in tracks
        print(f"Warning: GID {global_id} in mapping but missing in tracks.")
        return (255, 0, 0)  # Red for errors

    def get_triangulated_point(self, global_id: ObjectID) -> Optional[FloatArray]:
        """Return the last triangulated 3D point for a global ID, if any."""
        if global_id not in self.global_id_to_class:
            return None

        cls_id = self.global_id_to_class[global_id]
        if cls_id in self.global_tracks and global_id in self.global_tracks[cls_id]:
            return self.global_tracks[cls_id][global_id].triangulated_point

        print(f"Warning: GID {global_id} in mapping but missing in tracks.")
        return None

    def update_embedding(
        self, global_id: ObjectID, new_embedding: FloatArray, alpha: Optional[float] = None
    ) -> None:
        """
        Blend the existing embedding with a new one: e := α·e_old + (1-α)·e_new.
        Uses Config.EMBEDDING_ALPHA when alpha is None.
        """
        if alpha is None:
            alpha = Config.EMBEDDING_ALPHA
        if not isinstance(new_embedding, np.ndarray):
            raise TypeError("new_embedding must be a numpy array.")

        if global_id not in self.global_id_to_class:
            return

        cls_id = self.global_id_to_class[global_id]
        if cls_id not in self.global_tracks or global_id not in self.global_tracks[cls_id]:
            return

        track = self.global_tracks[cls_id][global_id]
        blended = alpha * track.embedding + (1.0 - alpha) * new_embedding
        norm = float(np.linalg.norm(blended))
        if norm > 0.0:
            blended /= norm  # Re-normalize to unit length
        self.global_tracks[cls_id][global_id].embedding = blended

    def triangulate_multiview_svd(
        self, positions: Dict[CameraId, Position], current_frame_number: int
    ) -> Optional[FloatArray]:
        """
        Triangulate a world-space 3D point using DLT + SVD from multiple views.

        Args:
            positions: Per-camera *undistorted* pixel points (recent only).
        Returns:
            (3,) float array [X, Y, Z] or None if insufficient/ill-posed.
        """
        rows: List[FloatArray] = []
        valid_views = 0

        for cam_id, pos in positions.items():
            if current_frame_number - pos.last_frame_updated > 10:
                continue  # Ignore stale observations

            calib = self.calibration_data.get(cam_id)
            if calib is None:
                continue

            # Projection: P = K [R | t]
            P = calib.K @ np.hstack([calib.R, calib.t])

            u, v = pos.point.x, pos.point.y
            p1T, p2T, p3T = P[0, :], P[1, :], P[2, :]

            # Two equations per view
            rows.append(u * p3T - p1T)
            rows.append(v * p3T - p2T)
            valid_views += 1

        if valid_views < 2:
            return None  # Need at least two cameras

        A = np.vstack(rows)  # Shape: (2*views, 4)
        # Solve Ax = 0 via SVD → last row of V^T is the solution
        _, _, Vh = np.linalg.svd(A)
        X_h = Vh[-1, :]

        if abs(X_h[3]) < 1e-12:
            # Homogeneous W≈0 → point at infinity
            print("Warning: Triangulation yielded W≈0 (point at infinity).")
            return None

        X = X_h[:3] / X_h[3]
        return X

    def update_position(
        self, global_id: ObjectID, camera_id: CameraId, bbox_center: Point, frame_number: int
    ) -> None:
        """
        Update last known 2D position for a global ID in a specific camera,
        and refresh its triangulated 3D point from all recent views.
        """
        if global_id not in self.global_id_to_class:
            return

        cls_id = self.global_id_to_class[global_id]
        if cls_id not in self.global_tracks or global_id not in self.global_tracks[cls_id]:
            return

        self.global_tracks[cls_id][global_id].positions[camera_id] = Position(
            bbox_center, frame_number
        )
        tri = self.triangulate_multiview_svd(
            self.global_tracks[cls_id][global_id].positions, frame_number
        )
        self.global_tracks[cls_id][global_id].triangulated_point = tri

    def reset(self) -> None:
        """Clear all state and caches; reset ID counter."""
        self.global_tracks.clear()
        self.global_id_to_class.clear()
        self.fundamental_matrices.clear()
        self.next_global_id = 1

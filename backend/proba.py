from ultralytics import YOLO
import cv2
import numpy as np
import json


def undistort_points(points, K, dist_coef):
    points = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
    undistorted = cv2.undistortPoints(points, K, dist_coef, P=K)
    return undistorted.reshape(-1, 2)


with open("calibration/calibration.json", 'r') as f:
    data = json.load(f)

camera_1 = data['cameras'][0]
camera_2 = data['cameras'][3]

K1 = np.array(camera_1['K'])
K2 = np.array(camera_2['K'])
R1 = np.array(camera_1['R'])
R2 = np.array(camera_2['R'])
t1 = np.array(camera_1['t']).flatten()
t2 = np.array(camera_2['t']).flatten()
dist_coef1 = np.array(camera_1["distCoef"])
dist_coef2 = np.array(camera_2["distCoef"])


def compute_fundamental_matrix():

    # Correct relative pose computation
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
    return F


def compute_epipolar_line(F, point):
    """
    Compute the epipolar line for a given point in the other camera view.

    :param F: Fundamental matrix
    :param point: 2D point (x, y) in homogeneous coordinates
    :return: Epipolar line equation [a, b, c] (ax + by + c = 0)
    """
    point_h = np.array([point[0], point[1], 1]
                       )  # Convert to homogeneous coordinates
    line = F @ point_h  # Compute epipolar line equation
    return line / np.linalg.norm(line[:2])  # Normalize line


def distance_point_to_line(point, line):
    """
    Compute distance of a point from a line.

    :param point: (x, y) coordinates of the point
    :param line: Epipolar line equation [a, b, c]
    :return: Perpendicular distance from point to line
    """
    return abs(line[0] * point[0] + line[1] * point[1] + line[2])


def find_correspondences(detections_cam1, detections_cam2, F, threshold=5):
    matches = []
    used_p2 = set()

    # Cam1 → Cam2
    for i, p1 in enumerate(detections_cam1):
        line = compute_epipolar_line(F, p1)
        best_dist = threshold
        best_j = -1
        for j, p2 in enumerate(detections_cam2):
            dist = distance_point_to_line(p2, line)
            if dist < best_dist and j not in used_p2:
                best_dist = dist
                best_j = j
        if best_j != -1:
            matches.append((i, best_j))
            used_p2.add(best_j)

    return matches


# Load YOLO model
yolo1 = YOLO('yolo12n.pt', verbose=False)
yolo2 = YOLO('yolo12n.pt', verbose=False)

# Video paths
video_path1 = "videos/hd_00_00.mp4"
video_path2 = "videos/hd_00_03.mp4"

# Open video files
cap1 = cv2.VideoCapture(video_path1)
cap2 = cv2.VideoCapture(video_path2)

frame_count = 0

while cap1.isOpened() and cap2.isOpened():
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()

    if not ret1 or not ret2:
        break

    frame_count += 1
    if frame_count % 5 != 0:
        continue

    # Run YOLO tracking
    results1 = yolo1.track(frame1, verbose=False)
    results2 = yolo2.track(frame2, verbose=False)

    # Get detections with tracking IDs
    detections_cam1 = [(box.xyxy[0].tolist(), int(box.id.item()) if box.id is not None else -1)
                       for box in results1[0].boxes]
    detections_cam2 = [(box.xyxy[0].tolist(), int(box.id.item()) if box.id is not None else -1)
                       for box in results2[0].boxes]

    # Extract centers and undistort
    detections_cam1_centers = [
        ((box[0]+box[2])/2, (box[1]+box[3])/2) for box, _ in detections_cam1]
    detections_cam2_centers = [
        ((box[0]+box[2])/2, (box[1]+box[3])/2) for box, _ in detections_cam2]

    detections_cam1_undistorted = undistort_points(
        detections_cam1_centers, K1, dist_coef1)
    detections_cam2_undistorted = undistort_points(
        detections_cam2_centers, K2, dist_coef2)

    # Compute matches
    F = compute_fundamental_matrix()
    matches = find_correspondences(
        detections_cam1_undistorted, detections_cam2_undistorted, F)

    # Assign random colors to matches
    match_colors = {}
    for idx1, idx2 in matches:
        color = tuple(np.random.randint(
            0, 255, 3).tolist())  # Random BGR color
        match_colors[idx1] = color
        match_colors[idx2 + 1000] = color  # Offset to avoid index collision

    # Draw bounding boxes with matched colors
    for i, (box, obj_id) in enumerate(detections_cam1):
        x1, y1, x2, y2 = map(int, box)
        color = match_colors.get(i, (0, 255, 0))  # Default green if no match
        cv2.rectangle(frame1, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame1, f"ID:{obj_id}", (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    for i, (box, obj_id) in enumerate(detections_cam2):
        x1, y1, x2, y2 = map(int, box)
        color = match_colors.get(i + 1000, (0, 255, 0))  # Offset index
        cv2.rectangle(frame2, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame2, f"ID:{obj_id}", (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Combine frames side-by-side
    combined = np.hstack((frame1, frame2))
    cv2.imshow("Multi-Camera Matching", combined)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap1.release()
cap2.release()
cv2.destroyAllWindows()

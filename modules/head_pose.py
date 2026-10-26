import cv2
import numpy as np


def get_head_pose(face_landmarks, frame_width, frame_height):

    image_points = np.array([
        (
            face_landmarks[1].x * frame_width,
            face_landmarks[1].y * frame_height
        ),
        (
            face_landmarks[152].x * frame_width,
            face_landmarks[152].y * frame_height
        ),
        (
            face_landmarks[33].x * frame_width,
            face_landmarks[33].y * frame_height
        ),
        (
            face_landmarks[263].x * frame_width,
            face_landmarks[263].y * frame_height
        ),
        (
            face_landmarks[61].x * frame_width,
            face_landmarks[61].y * frame_height
        ),
        (
            face_landmarks[291].x * frame_width,
            face_landmarks[291].y * frame_height
        )
    ], dtype=np.float64)

    model_points = np.array([
        (0.0, 0.0, 0.0),
        (0.0, -63.6, -12.5),
        (-43.3, 32.7, -26.0),
        (43.3, 32.7, -26.0),
        (-28.9, -28.9, -24.1),
        (28.9, -28.9, -24.1)
    ], dtype=np.float64)

    focal_length = frame_width

    camera_matrix = np.array([
        [focal_length, 0, frame_width / 2],
        [0, focal_length, frame_height / 2],
        [0, 0, 1]
    ], dtype=np.float64)

    distortion = np.zeros((4, 1))

    success, rotation_vector, translation_vector = cv2.solvePnP(
        model_points,
        image_points,
        camera_matrix,
        distortion,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not success:
        return None

    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rotation_matrix)

    pitch = angles[0]
    yaw = angles[1]
    roll = angles[2]

    return pitch, yaw, roll


def is_head_turned(yaw, threshold=25):
    return abs(yaw) > threshold
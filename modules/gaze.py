import numpy as np


def get_point(landmarks, index, width, height):

    return np.array([
        landmarks[index].x * width,
        landmarks[index].y * height
    ])


def get_gaze_ratio(landmarks, eye_indices, iris_indices, width, height):

    eye_points = [
        get_point(landmarks, index, width, height)
        for index in eye_indices
    ]

    iris_points = [
        get_point(landmarks, index, width, height)
        for index in iris_indices
    ]

    eye_left = min(point[0] for point in eye_points)
    eye_right = max(point[0] for point in eye_points)

    iris_center = np.mean(iris_points, axis=0)

    eye_width = eye_right - eye_left

    if eye_width == 0:
        return 0.5

    return (iris_center[0] - eye_left) / eye_width


def get_gaze_direction(left_ratio, right_ratio):

    average_ratio = (left_ratio + right_ratio) / 2

    if average_ratio < 0.39:
        return "RIGHT"

    if average_ratio > 0.59:
        return "LEFT"

    return "CENTER"
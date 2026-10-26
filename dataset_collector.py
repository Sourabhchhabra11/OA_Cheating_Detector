import cv2
import mediapipe as mp
import csv
import os
from datetime import datetime

from modules.head_pose import get_head_pose
from modules.gaze import get_gaze_ratio, get_gaze_direction
from modules.phone_detector import PhoneDetector


BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


options = FaceLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="models/face_landmarker.task"
    ),
    running_mode=VisionRunningMode.VIDEO,
    num_faces=2
)


os.makedirs("dataset", exist_ok=True)

csv_file = "dataset/oa_features.csv"

file_exists = os.path.exists(csv_file)

file = open(
    csv_file,
    "a",
    newline=""
)

writer = csv.writer(file)

if not file_exists:

    writer.writerow([
        "timestamp",
        "face_count",
        "yaw",
        "pitch",
        "roll",
        "left_gaze",
        "right_gaze",
        "gaze_direction",
        "phone_present",
        "label"
    ])


cap = cv2.VideoCapture(0)

phone_detector = PhoneDetector()

current_label = None

recording = False

timestamp = 0

print()
print("======================================")
print("       OA DATASET COLLECTOR")
print("======================================")
print()
print("1 = NORMAL")
print("2 = LOOKING AWAY")
print("3 = PHONE")
print("4 = MULTIPLE FACES")
print("5 = COMBINED")
print()
print("SPACE = START / STOP RECORDING")
print("Q = QUIT")
print()
print("First select a label.")
print()


with FaceLandmarker.create_from_options(options) as landmarker:

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        h, w, _ = frame.shape

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        timestamp += 1

        result = landmarker.detect_for_video(
            mp_image,
            timestamp
        )

        face_count = len(
            result.face_landmarks
        )

        phones = phone_detector.detect(frame)

        phone_present = 1 if len(phones) > 0 else 0

        for phone in phones:

            x1, y1, x2, y2 = phone["box"]

            confidence = phone["confidence"]

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 0, 255),
                2
            )

            cv2.putText(
                frame,
                f"Phone {confidence:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2
            )

        yaw = 0.0
        pitch = 0.0
        roll = 0.0

        left_gaze = 0.5
        right_gaze = 0.5

        gaze_direction = "NO_FACE"

        if face_count > 0:

            face_landmarks = result.face_landmarks[0]

            for landmark in face_landmarks:

                x = int(landmark.x * w)
                y = int(landmark.y * h)

                cv2.circle(
                    frame,
                    (x, y),
                    1,
                    (0, 255, 0),
                    -1
                )

            left_gaze = get_gaze_ratio(
                face_landmarks,
                [33, 133, 159, 145],
                [469, 470, 471, 472],
                w,
                h
            )

            right_gaze = get_gaze_ratio(
                face_landmarks,
                [362, 263, 386, 374],
                [474, 475, 476, 477],
                w,
                h
            )

            gaze_direction = get_gaze_direction(
                left_gaze,
                right_gaze
            )

            pose = get_head_pose(
                face_landmarks,
                w,
                h
            )

            if pose:

                pitch, yaw, roll = pose

        if recording and current_label is not None:

            writer.writerow([
                datetime.now().isoformat(),
                face_count,
                round(yaw, 2),
                round(pitch, 2),
                round(roll, 2),
                round(left_gaze, 3),
                round(right_gaze, 3),
                gaze_direction,
                phone_present,
                current_label
            ])

            file.flush()

        cv2.putText(
            frame,
            f"Faces: {face_count}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Yaw: {yaw:.1f}",
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Pitch: {pitch:.1f}",
            (20, 105),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Roll: {roll:.1f}",
            (20, 135),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Gaze: {gaze_direction}",
            (20, 170),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Phone: {phone_present}",
            (20, 205),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2
        )

        if current_label is None:

            label_text = "LABEL: NOT SELECTED"

        else:

            label_text = f"LABEL: {current_label}"

        cv2.putText(
            frame,
            label_text,
            (20, 250),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

        if recording:

            status = "RECORDING"

        else:

            status = "PAUSED"

        cv2.putText(
            frame,
            status,
            (20, 285),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv2.imshow(
            "OA Dataset Collector",
            frame
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("1"):

            current_label = "normal"
            recording = False

            print("LABEL SELECTED: NORMAL")
            print("Press SPACE to start recording.")

        elif key == ord("2"):

            current_label = "looking_away"
            recording = False

            print("LABEL SELECTED: LOOKING_AWAY")
            print("Press SPACE to start recording.")

        elif key == ord("3"):

            current_label = "phone"
            recording = False

            print("LABEL SELECTED: PHONE")
            print("Press SPACE to start recording.")

        elif key == ord("4"):

            current_label = "multiple_faces"
            recording = False

            print("LABEL SELECTED: MULTIPLE_FACES")
            print("Press SPACE to start recording.")

        elif key == ord("5"):

            current_label = "combined"
            recording = False

            print("LABEL SELECTED: COMBINED")
            print("Press SPACE to start recording.")

        elif key == ord(" ") and current_label is not None:

            recording = not recording

            if recording:

                print(
                    f"RECORDING STARTED: {current_label}"
                )

            else:

                print(
                    f"RECORDING STOPPED: {current_label}"
                )

        elif key == ord("q"):

            break


file.close()

cap.release()

cv2.destroyAllWindows()

print()
print("======================================")
print("DATA COLLECTION FINISHED")
print("======================================")
print()
print("Dataset:")
print(csv_file)
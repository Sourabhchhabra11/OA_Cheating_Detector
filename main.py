import cv2
import time
import joblib
import mediapipe as mp
import pandas as pd

from modules.head_pose import get_head_pose

from modules.behavior_tracker import (
    GazeTracker,
    HeadTurnTracker,
    MultipleFaceTracker,
    PhoneTracker
)

from modules.gaze import (
    get_gaze_ratio,
    get_gaze_direction
)

from modules.phone_detector import PhoneDetector
from modules.suspicion_engine import SuspicionEngine
from modules.event_logger import EventLogger


looking_away_model = joblib.load(
    "models/looking_away_model.pkl"
)

phone_model = joblib.load(
    "models/phone_model.pkl"
)

multiple_faces_model = joblib.load(
    "models/multiple_faces_model.pkl"
)

behavior_preprocessor = joblib.load(
    "models/behavior_preprocessor.pkl"
)


BaseOptions = mp.tasks.BaseOptions

FaceLandmarker = (
    mp.tasks.vision.FaceLandmarker
)

FaceLandmarkerOptions = (
    mp.tasks.vision.FaceLandmarkerOptions
)

VisionRunningMode = (
    mp.tasks.vision.RunningMode
)


options = FaceLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="models/face_landmarker.task"
    ),
    running_mode=VisionRunningMode.VIDEO,
    num_faces=2
)


cap = cv2.VideoCapture(0)


gaze_tracker = GazeTracker(
    duration=1.5
)

head_tracker = HeadTurnTracker(
    threshold=17,
    duration=1.5
)

multiple_face_tracker = MultipleFaceTracker(
    duration=1.5
)

phone_tracker = PhoneTracker(
    duration=1.5
)

phone_detector = PhoneDetector()

suspicion_engine = SuspicionEngine()

event_logger = EventLogger()


frame_count = 0

ml_looking_away = 0
ml_phone = 0
ml_multiple_faces = 0
ml_multiple_faces_probability = 0.0


with FaceLandmarker.create_from_options(
    options
) as landmarker:

    timestamp = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame_count += 1

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

        phones = phone_detector.detect(
            frame
        )

        phone_detected = (
            len(phones) > 0
        )

        multiple_faces_active = (
            multiple_face_tracker.update(
                face_count
            )
        )

        looking_away_active = False

        gaze_active = False
        head_active = False

        left_ratio = 0.5
        right_ratio = 0.5

        gaze_direction = "CENTER"

        pitch = 0.0
        yaw = 0.0
        roll = 0.0


        for phone in phones:

            x1, y1, x2, y2 = (
                phone["box"]
            )

            confidence = (
                phone["confidence"]
            )

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


        for face_landmarks in result.face_landmarks:

            for landmark in face_landmarks:

                x = int(
                    landmark.x * w
                )

                y = int(
                    landmark.y * h
                )

                cv2.circle(
                    frame,
                    (x, y),
                    1,
                    (0, 255, 0),
                    -1
                )


            left_ratio = get_gaze_ratio(
                face_landmarks,
                [33, 133, 159, 145],
                [469, 470, 471, 472],
                w,
                h
            )

            right_ratio = get_gaze_ratio(
                face_landmarks,
                [362, 263, 386, 374],
                [474, 475, 476, 477],
                w,
                h
            )


            gaze_direction = get_gaze_direction(
                left_ratio,
                right_ratio
            )


            gaze_active = (
                gaze_tracker.update(
                    gaze_direction
                )
            )


            pose = get_head_pose(
                face_landmarks,
                w,
                h
            )


            if pose:

                pitch, yaw, roll = pose

                head_active = (
                    head_tracker.update(
                        yaw
                    )
                )

            break


        if gaze_direction != "CENTER":

            if (
                gaze_tracker.start_time
                is not None
            ):

                gaze_elapsed = (
                    time.time()
                    - gaze_tracker.start_time
                )

                cv2.putText(
                    frame,
                    f"Gaze Timer: {gaze_elapsed:.1f}s",
                    (400, 160),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2
                )


        if abs(yaw) > 10:

            if (
                head_tracker.start_time
                is not None
            ):

                head_elapsed = (
                    time.time()
                    - head_tracker.start_time
                )

                cv2.putText(
                    frame,
                    f"Head Timer: {head_elapsed:.1f}s",
                    (400, 190),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2
                )


        if gaze_active or head_active:

            looking_away_active = True


        if frame_count % 10 == 0:

            model_input = pd.DataFrame(
                [[
                    face_count,
                    yaw,
                    pitch,
                    roll,
                    left_ratio,
                    right_ratio,
                    gaze_direction,
                    int(phone_detected)
                ]],
                columns=[
                    "face_count",
                    "yaw",
                    "pitch",
                    "roll",
                    "left_gaze",
                    "right_gaze",
                    "gaze_direction",
                    "phone_present"
                ]
            )


            model_features = (
                behavior_preprocessor.transform(
                    model_input
                )
            )


            ml_looking_away = int(
                looking_away_model.predict(
                    model_features
                )[0]
            )


            ml_phone = int(
                phone_model.predict(
                    model_features
                )[0]
            )


            ml_multiple_faces_probability = (
                multiple_faces_model
                .predict_proba(
                    model_features
                )[0][1]
            )


            ml_multiple_faces = int(
                ml_multiple_faces_probability >= 0.70
            )


        previous_event_count = len(
            suspicion_engine.get_events()
        )


        suspicion_score, status = (
            suspicion_engine.update(
                looking_away_active,
                phone_detected,
                multiple_faces_active
            )
        )


        current_events = (
            suspicion_engine.get_events()
        )


        if (
            len(current_events)
            > previous_event_count
        ):

            new_event = current_events[-1]

            event_logger.log_event(
                new_event["event"],
                new_event["score"]
            )


        if phone_detected:

            cv2.putText(
                frame,
                "PHONE DETECTED",
                (20, 370),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )


        if face_count > 1:

            cv2.putText(
                frame,
                "MULTIPLE FACES",
                (20, 330),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )


        if looking_away_active:

            cv2.putText(
                frame,
                "LOOKING AWAY",
                (20, 180),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )


        cv2.putText(
            frame,
            f"Faces: {face_count}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )


        cv2.putText(
            frame,
            f"Pitch: {pitch:.1f}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )


        cv2.putText(
            frame,
            f"Yaw: {yaw:.1f}",
            (20, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )


        cv2.putText(
            frame,
            f"Roll: {roll:.1f}",
            (20, 140),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )


        cv2.putText(
            frame,
            f"Left Eye: {left_ratio:.2f}",
            (20, 220),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2
        )


        cv2.putText(
            frame,
            f"Right Eye: {right_ratio:.2f}",
            (20, 250),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2
        )


        cv2.putText(
            frame,
            f"Gaze: {gaze_direction}",
            (20, 290),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 0),
            2
        )


        cv2.putText(
            frame,
            f"ML Looking Away: {ml_looking_away}",
            (650, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )


        cv2.putText(
            frame,
            f"ML Phone: {ml_phone}",
            (650, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )


        cv2.putText(
            frame,
            f"ML Multiple Faces: "
            f"{ml_multiple_faces}",
            (650, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )


        cv2.putText(
            frame,
            f"MF Probability: "
            f"{ml_multiple_faces_probability:.2f}",
            (650, 130),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )


        cv2.putText(
            frame,
            f"Gaze Away: "
            f"{'YES' if gaze_active else 'NO'}",
            (400, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255)
            if gaze_active
            else (0, 255, 0),
            2
        )


        cv2.putText(
            frame,
            f"Head Turn: "
            f"{'YES' if head_active else 'NO'}",
            (400, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255)
            if head_active
            else (0, 255, 0),
            2
        )


        cv2.putText(
            frame,
            f"Phone: "
            f"{'YES' if phone_detected else 'NO'}",
            (400, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255)
            if phone_detected
            else (0, 255, 0),
            2
        )


        cv2.putText(
            frame,
            f"Multiple Faces: "
            f"{'YES' if multiple_faces_active else 'NO'}",
            (400, 130),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255)
            if multiple_faces_active
            else (0, 255, 0),
            2
        )


        cv2.putText(
            frame,
            f"Suspicion Score: "
            f"{suspicion_score}",
            (20, 420),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )


        cv2.putText(
            frame,
            f"Status: {status}",
            (20, 460),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 255)
            if suspicion_score > 0
            else (0, 255, 0),
            2
        )


        cv2.imshow(
            "OA Cheating Detector",
            frame
        )


        if (
            cv2.waitKey(1) & 0xFF
            == ord("q")
        ):

            break


cap.release()

cv2.destroyAllWindows()

event_logger.print_summary(suspicion_engine.score)
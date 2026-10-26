import cv2
import time
import threading
import joblib
import mediapipe as mp
import pandas as pd
import inspect

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


class MonitoringService:

    def __init__(self):

        self.running = False
        self.thread = None
        self.cap = None

        self.frame = None
        self.jpeg_frame = None

        self.frame_lock = threading.Lock()
        self.status_lock = threading.Lock()

        # Dashboard encoding is intentionally capped separately from inference.
        self.last_publish_time = 0.0
        self.publish_interval = 1.0 / 15.0

        self.last_logged_event_count = 0
        self.session_start_time = None

        self.status_data = self._default_status()

        self.looking_away_model = joblib.load(
            "models/looking_away_model.pkl"
        )

        self.phone_model = joblib.load(
            "models/phone_model.pkl"
        )

        self.multiple_faces_model = joblib.load(
            "models/multiple_faces_model.pkl"
        )

        self.behavior_preprocessor = joblib.load(
            "models/behavior_preprocessor.pkl"
        )

        self.phone_detector = PhoneDetector()
        self.suspicion_engine = SuspicionEngine()
        self.event_logger = EventLogger()

        self._reset_trackers()

    def _default_status(self):

        return {
            "running": False,
            "session_duration": 0,
            "elapsed_seconds": 0,
            "started_at": None,
            "face_count": 0,
            "pitch": 0.0,
            "yaw": 0.0,
            "roll": 0.0,
            "left_gaze": 0.5,
            "right_gaze": 0.5,
            "gaze_direction": "CENTER",
            "gaze_active": False,
            "head_active": False,
            "phone_detected": False,
            "multiple_faces": False,
            "ml_looking_away": 0,
            "ml_phone": 0,
            "ml_multiple_faces": 0,
            "ml_multiple_faces_probability": 0.0,
            "suspicion_score": 0,
            "status": "NORMAL",
            "events": []
        }

    def _reset_trackers(self):

        self.gaze_tracker = GazeTracker(
            duration=1.5
        )

        self.head_tracker = HeadTurnTracker(
            threshold=17,
            duration=1.5
        )

        self.multiple_face_tracker = MultipleFaceTracker(
            duration=1.5
        )

        self.phone_tracker = PhoneTracker(
            duration=1.5
        )

        self.frame_count = 0

        self.ml_looking_away = 0
        self.ml_phone = 0
        self.ml_multiple_faces = 0
        self.ml_multiple_faces_probability = 0.0

    def _encode_frame(self, frame):

        success, buffer = cv2.imencode(
            ".jpg",
            frame,
            [
                cv2.IMWRITE_JPEG_QUALITY,
                72
            ]
        )

        if not success:
            return None

        return buffer.tobytes()

    def _publish_frame(self, frame, force=False):

        # JPEG encoding is CPU-heavy. Limit only browser-frame encoding;
        # detection and event processing continue at the fastest available rate.
        now = time.monotonic()
        if not force and now - self.last_publish_time < self.publish_interval:
            return

        jpeg = self._encode_frame(frame)
        if jpeg is None:
            return

        with self.frame_lock:
            self.frame = frame.copy()
            self.jpeg_frame = jpeg

        self.last_publish_time = now

    def start(self):

        if self.running:
            return False, "Monitoring is already running."

        self.cap = cv2.VideoCapture(
            0,
            cv2.CAP_DSHOW
        )

        if not self.cap.isOpened():

            self.cap.release()

            self.cap = cv2.VideoCapture(0)

        if not self.cap.isOpened():

            self.cap.release()
            self.cap = None

            return False, "Could not open camera."

        self.cap.set(
            cv2.CAP_PROP_FRAME_WIDTH,
            640
        )

        self.cap.set(
            cv2.CAP_PROP_FRAME_HEIGHT,
            480
        )

        self.cap.set(
            cv2.CAP_PROP_FPS,
            30
        )

        # Keep the camera queue short so processing works on recent frames.
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        ok, first_frame = self.cap.read()

        if not ok or first_frame is None:

            self.cap.release()
            self.cap = None

            return False, "Camera opened but no frame could be captured."

        self.last_publish_time = 0.0
        self._publish_frame(first_frame, force=True)

        self.session_start_time = time.monotonic()
        self.running = True

        with self.status_lock:
            self.status_data["running"] = True

        self.thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True
        )

        self.thread.start()

        return True, "Monitoring started."

    def stop(self):

        self.running = False
        self.session_start_time = None
        self.last_publish_time = 0.0

        if self.thread is not None:

            self.thread.join(
                timeout=3
            )

        self.thread = None

        if self.cap is not None:

            self.cap.release()
            self.cap = None

        with self.frame_lock:

            self.frame = None
            self.jpeg_frame = None

        with self.status_lock:

            self.status_data["running"] = False

        try:

            self.event_logger.print_summary(
                self.suspicion_engine.score
            )

        except Exception as exc:

            print(
                "Could not print summary:",
                exc
            )

    def reset(self):

        self.stop()
        self.session_start_time = None

        self.suspicion_engine = SuspicionEngine()
        self.event_logger = EventLogger()

        self.last_logged_event_count = 0

        self._reset_trackers()

        with self.frame_lock:

            self.frame = None
            self.jpeg_frame = None

        with self.status_lock:

            self.status_data = self._default_status()

    def get_status(self):

        with self.status_lock:

            data = dict(
                self.status_data
            )

            data["events"] = list(
                self.suspicion_engine.events
            )

            if self.running and self.session_start_time is not None:

                elapsed = int(
                    time.monotonic()
                    - self.session_start_time
                )

            else:

                elapsed = int(
                    data.get(
                        "session_duration",
                        0
                    )
                )

            data["session_duration"] = elapsed
            data["elapsed_seconds"] = elapsed
            data["duration"] = elapsed

            if self.session_start_time is not None:

                data["started_at"] = (
                    time.time()
                    - (
                        time.monotonic()
                        - self.session_start_time
                    )
                )

            else:

                data["started_at"] = None

            return data

    def get_jpeg(self):

        with self.frame_lock:

            if self.jpeg_frame is None:
                return None

            return self.jpeg_frame

    def _log_new_events(self):

        try:

            events = getattr(
                self.suspicion_engine,
                "events",
                []
            )

            if not isinstance(events, list):
                return

            while self.last_logged_event_count < len(events):

                event = events[
                    self.last_logged_event_count
                ]

                self.last_logged_event_count += 1

                if isinstance(event, dict):

                    event_name = event.get(
                        "event",
                        "UNKNOWN"
                    )

                    event_score = event.get(
                        "score",
                        self.suspicion_engine.score
                    )

                else:

                    event_name = str(event)

                    event_score = (
                        self.suspicion_engine.score
                    )

                try:

                    signature = inspect.signature(
                        self.event_logger.log_event
                    )

                    parameters = [
                        p
                        for p in signature.parameters.values()
                        if p.name != "self"
                    ]

                    if len(parameters) >= 3:

                        self.event_logger.log_event(
                            time.strftime(
                                "%Y-%m-%d %H:%M:%S"
                            ),
                            event_name,
                            event_score
                        )

                    else:

                        self.event_logger.log_event(
                            event_name,
                            event_score
                        )

                except Exception as exc:

                    print(
                        "Could not log event:",
                        exc
                    )

        except Exception as exc:

            print(
                "Event processing error:",
                exc
            )

    def _create_landmarker(self):

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
                model_asset_path=
                "models/face_landmarker.task"
            ),

            running_mode=
            VisionRunningMode.VIDEO,

            num_faces=2
        )

        return FaceLandmarker.create_from_options(
            options
        )

    def _monitor_loop(self):

        timestamp = 0

        try:

            with self._create_landmarker() as landmarker:

                while self.running:

                    if self.cap is None:
                        break

                    ret, frame = self.cap.read()

                    if not ret or frame is None:

                        time.sleep(0.01)
                        continue

                    self.frame_count += 1

                    h, w, _ = frame.shape

                    try:

                        rgb = cv2.cvtColor(
                            frame,
                            cv2.COLOR_BGR2RGB
                        )

                        mp_image = mp.Image(
                            image_format=
                            mp.ImageFormat.SRGB,
                            data=rgb
                        )

                        new_timestamp = int(
                            time.monotonic() * 1000
                        )

                        if new_timestamp <= timestamp:
                            new_timestamp = timestamp + 1

                        timestamp = new_timestamp

                        result = (
                            landmarker.detect_for_video(
                                mp_image,
                                timestamp
                            )
                        )

                    except Exception as exc:

                        cv2.putText(
                            frame,
                            "PROCESSING ERROR",
                            (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 0, 255),
                            2
                        )

                        print(
                            "MediaPipe processing error:",
                            exc
                        )

                        self._publish_frame(frame)
                        continue

                    face_count = len(
                        result.face_landmarks
                    )

                    try:

                        phones = (
                            self.phone_detector.detect(
                                frame
                            )
                        )

                    except Exception as exc:

                        print(
                            "Phone detector error:",
                            exc
                        )

                        phones = []

                    phone_detected = (
                        len(phones) > 0
                    )

                    multiple_faces_active = (
                        self.multiple_face_tracker.update(
                            face_count
                        )
                    )

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
                            (x1, max(20, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 0, 255),
                            2
                        )

                    for face_landmarks in result.face_landmarks:

                        # Drawing all 478 landmarks on every frame is expensive.
                        # Draw only the key points used by gaze/head analysis.
                        key_points = (33, 133, 159, 145, 469, 470, 471, 472,
                                      362, 263, 386, 374, 474, 475, 476, 477,
                                      1, 10, 152, 234, 454)
                        for point_index in key_points:
                            if point_index < len(face_landmarks):
                                landmark = face_landmarks[point_index]
                                x = int(landmark.x * w)
                                y = int(landmark.y * h)
                                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

                        try:

                            left_ratio = (
                                get_gaze_ratio(
                                    face_landmarks,
                                    [33, 133, 159, 145],
                                    [469, 470, 471, 472],
                                    w,
                                    h
                                )
                            )

                            right_ratio = (
                                get_gaze_ratio(
                                    face_landmarks,
                                    [362, 263, 386, 374],
                                    [474, 475, 476, 477],
                                    w,
                                    h
                                )
                            )

                            gaze_direction = (
                                get_gaze_direction(
                                    left_ratio,
                                    right_ratio
                                )
                            )

                            gaze_active = (
                                self.gaze_tracker.update(
                                    gaze_direction
                                )
                            )

                        except Exception as exc:

                            print(
                                "Gaze processing error:",
                                exc
                            )

                        try:

                            pose = get_head_pose(
                                face_landmarks,
                                w,
                                h
                            )

                            if pose:

                                pitch, yaw, roll = pose

                                head_active = (
                                    self.head_tracker.update(
                                        yaw
                                    )
                                )

                        except Exception as exc:

                            print(
                                "Head pose error:",
                                exc
                            )

                        break

                    looking_away_active = (
                        gaze_active
                        or head_active
                    )

                    if gaze_direction != "CENTER":

                        if (
                            self.gaze_tracker.start_time
                            is not None
                        ):

                            gaze_elapsed = (
                                time.time()
                                - self.gaze_tracker.start_time
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
                            self.head_tracker.start_time
                            is not None
                        ):

                            head_elapsed = (
                                time.time()
                                - self.head_tracker.start_time
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

                    if self.frame_count % 10 == 0:

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

                        try:

                            model_features = (
                                self.behavior_preprocessor.transform(
                                    model_input
                                )
                            )

                            self.ml_looking_away = int(
                                self.looking_away_model.predict(
                                    model_features
                                )[0]
                            )

                            self.ml_phone = int(
                                self.phone_model.predict(
                                    model_features
                                )[0]
                            )

                            self.ml_multiple_faces_probability = (
                                self.multiple_faces_model
                                .predict_proba(
                                    model_features
                                )[0][1]
                            )

                            self.ml_multiple_faces = int(
                                self.ml_multiple_faces_probability
                                >= 0.70
                            )

                        except Exception as exc:

                            print(
                                "ML prediction error:",
                                exc
                            )

                    suspicion_score, status = (
                        self.suspicion_engine.update(
                            looking_away_active,
                            phone_detected,
                            multiple_faces_active
                        )
                    )

                    self._log_new_events()

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
                        f"ML Looking Away: {self.ml_looking_away}",
                        (360, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2
                    )

                    cv2.putText(
                        frame,
                        f"ML Phone: {self.ml_phone}",
                        (360, 70),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2
                    )

                    cv2.putText(
                        frame,
                        f"ML Multiple Faces: {self.ml_multiple_faces}",
                        (360, 100),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2
                    )

                    cv2.putText(
                        frame,
                        f"MF Probability: "
                        f"{self.ml_multiple_faces_probability:.2f}",
                        (360, 130),
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

                    with self.status_lock:

                        self.status_data.update({

                            "running": True,

                            "session_duration": (
                                int(
                                    time.monotonic()
                                    - self.session_start_time
                                )
                                if self.session_start_time is not None
                                else 0
                            ),

                            "elapsed_seconds": (
                                int(
                                    time.monotonic()
                                    - self.session_start_time
                                )
                                if self.session_start_time is not None
                                else 0
                            ),

                            "face_count": face_count,

                            "pitch": round(
                                pitch,
                                2
                            ),

                            "yaw": round(
                                yaw,
                                2
                            ),

                            "roll": round(
                                roll,
                                2
                            ),

                            "left_gaze": round(
                                left_ratio,
                                3
                            ),

                            "right_gaze": round(
                                right_ratio,
                                3
                            ),

                            "gaze_direction":
                                gaze_direction,

                            "gaze_active":
                                gaze_active,

                            "head_active":
                                head_active,

                            "phone_detected":
                                phone_detected,

                            "multiple_faces":
                                multiple_faces_active,

                            "ml_looking_away":
                                self.ml_looking_away,

                            "ml_phone":
                                self.ml_phone,

                            "ml_multiple_faces":
                                self.ml_multiple_faces,

                            "ml_multiple_faces_probability":
                                round(
                                    self.ml_multiple_faces_probability,
                                    3
                                ),

                            "suspicion_score":
                                suspicion_score,

                            "status":
                                status,

                            "events":
                                list(
                                    self.suspicion_engine.events
                                )
                        })

                    # IMPORTANT:
                    # Only the fully processed frame is published here.
                    # The browser never receives a competing raw frame.
                    self._publish_frame(frame)

        except Exception as exc:

            print(
                "Monitoring loop error:",
                exc
            )

        finally:

            with self.status_lock:
                self.status_data["running"] = False

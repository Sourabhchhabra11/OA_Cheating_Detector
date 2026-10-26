
import os
import time
import threading
import cv2
import joblib
import mediapipe as mp
import pandas as pd
import streamlit as st
import av

from pathlib import Path
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

from modules.head_pose import get_head_pose
from modules.behavior_tracker import (
    GazeTracker,
    HeadTurnTracker,
    MultipleFaceTracker
)
from modules.gaze import get_gaze_ratio, get_gaze_direction
from modules.phone_detector import PhoneDetector
from modules.suspicion_engine import SuspicionEngine
from modules.event_logger import EventLogger


st.set_page_config(
    page_title="OA Cheating Detector",
    page_icon="🎓",
    layout="wide"
)


ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)

MODEL_DIR = ROOT / "models"
DATASET_DIR = ROOT / "dataset"

DATASET_DIR.mkdir(exist_ok=True)

looking_away_model = joblib.load(
    MODEL_DIR / "looking_away_model.pkl"
)

phone_model = joblib.load(
    MODEL_DIR / "phone_model.pkl"
)

multiple_faces_model = joblib.load(
    MODEL_DIR / "multiple_faces_model.pkl"
)

behavior_preprocessor = joblib.load(
    MODEL_DIR / "behavior_preprocessor.pkl"
)


class DetectionProcessor(VideoProcessorBase):

    def __init__(self):
        self.lock = threading.Lock()

        self.landmarker = None

        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_path=str(
                    MODEL_DIR / "face_landmarker.task"
                )
            ),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_faces=2
        )

        self.landmarker = (
            mp.tasks.vision.FaceLandmarker
            .create_from_options(options)
        )

        self.gaze_tracker = GazeTracker(duration=1.5)
        self.head_tracker = HeadTurnTracker(
            threshold=17,
            duration=1.5
        )
        self.multiple_face_tracker = MultipleFaceTracker(
            duration=1.5
        )

        self.phone_detector = PhoneDetector()
        self.suspicion_engine = SuspicionEngine()
        self.event_logger = EventLogger()

        self.frame_count = 0
        self.timestamp_ms = 0

        self.gaze_active = False
        self.head_active = False
        self.phone_detected = False
        self.multiple_faces_active = False
        self.looking_away_active = False

        self.face_count = 0
        self.gaze_direction = "CENTER"

        self.pitch = 0.0
        self.yaw = 0.0
        self.roll = 0.0

        self.left_ratio = 0.5
        self.right_ratio = 0.5

        self.ml_looking_away = 0
        self.ml_phone = 0
        self.ml_multiple_faces = 0
        self.ml_multiple_faces_probability = 0.0

        self.suspicion_score = 0
        self.status = "NORMAL"

        self.last_error = ""
        self.report_saved = False

    def recv(self, frame):

        image = frame.to_ndarray(format="bgr24")

        try:
            self.frame_count += 1

            height, width, _ = image.shape

            rgb = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2RGB
            )

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb
            )

            self.timestamp_ms += 34

            result = self.landmarker.detect_for_video(
                mp_image,
                self.timestamp_ms
            )

            face_count = len(result.face_landmarks)

            phones = self.phone_detector.detect(image)
            phone_detected = len(phones) > 0

            multiple_faces_active = (
                self.multiple_face_tracker.update(face_count)
            )

            left_ratio = 0.5
            right_ratio = 0.5
            gaze_direction = "CENTER"

            pitch = 0.0
            yaw = 0.0
            roll = 0.0

            gaze_active = False
            head_active = False

            for phone in phones:
                x1, y1, x2, y2 = phone["box"]
                confidence = phone["confidence"]

                cv2.rectangle(
                    image,
                    (x1, y1),
                    (x2, y2),
                    (0, 0, 255),
                    2
                )

                cv2.putText(
                    image,
                    f"Phone {confidence:.2f}",
                    (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2
                )

            for face_landmarks in result.face_landmarks:

                for landmark in face_landmarks:
                    x = int(landmark.x * width)
                    y = int(landmark.y * height)

                    cv2.circle(
                        image,
                        (x, y),
                        1,
                        (0, 255, 0),
                        -1
                    )

            if face_count > 0:

                face_landmarks = result.face_landmarks[0]

                left_ratio = get_gaze_ratio(
                    face_landmarks,
                    [33, 133, 159, 145],
                    [469, 470, 471, 472],
                    width,
                    height
                )

                right_ratio = get_gaze_ratio(
                    face_landmarks,
                    [362, 263, 386, 374],
                    [474, 475, 476, 477],
                    width,
                    height
                )

                gaze_direction = get_gaze_direction(
                    left_ratio,
                    right_ratio
                )

                gaze_active = self.gaze_tracker.update(
                    gaze_direction
                )

                pose = get_head_pose(
                    face_landmarks,
                    width,
                    height
                )

                if pose:
                    pitch, yaw, roll = pose
                    head_active = self.head_tracker.update(yaw)

            looking_away_active = (
                gaze_active or head_active
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

                model_features = (
                    behavior_preprocessor.transform(model_input)
                )

                self.ml_looking_away = int(
                    looking_away_model.predict(
                        model_features
                    )[0]
                )

                self.ml_phone = int(
                    phone_model.predict(
                        model_features
                    )[0]
                )

                self.ml_multiple_faces_probability = float(
                    multiple_faces_model.predict_proba(
                        model_features
                    )[0][1]
                )

                self.ml_multiple_faces = int(
                    self.ml_multiple_faces_probability >= 0.70
                )

            previous_event_count = len(
                self.suspicion_engine.get_events()
            )

            suspicion_score, status = (
                self.suspicion_engine.update(
                    looking_away_active,
                    phone_detected,
                    multiple_faces_active
                )
            )

            current_events = self.suspicion_engine.get_events()

            if len(current_events) > previous_event_count:
                for event in current_events[previous_event_count:]:
                    self.event_logger.log_event(
                        event["event"],
                        event["score"]
                    )

            self.face_count = face_count
            self.phone_detected = phone_detected
            self.multiple_faces_active = multiple_faces_active
            self.looking_away_active = looking_away_active

            self.gaze_active = gaze_active
            self.head_active = head_active

            self.gaze_direction = gaze_direction
            self.left_ratio = left_ratio
            self.right_ratio = right_ratio

            self.pitch = pitch
            self.yaw = yaw
            self.roll = roll

            self.suspicion_score = suspicion_score
            self.status = status

            self.last_error = ""

            labels = [
                (f"Faces: {face_count}", (0, 255, 0)),
                (f"Gaze: {gaze_direction}", (255, 255, 0)),
                (f"Head Turn: {'YES' if head_active else 'NO'}",
                 (0, 0, 255) if head_active else (0, 255, 0)),
                (f"Gaze Away: {'YES' if gaze_active else 'NO'}",
                 (0, 0, 255) if gaze_active else (0, 255, 0)),
                (f"Phone: {'YES' if phone_detected else 'NO'}",
                 (0, 0, 255) if phone_detected else (0, 255, 0)),
                (f"Multiple Faces: {'YES' if multiple_faces_active else 'NO'}",
                 (0, 0, 255) if multiple_faces_active else (0, 255, 0)),
                (f"Score: {suspicion_score}", (0, 255, 255)),
                (f"Status: {status}",
                 (0, 0, 255) if suspicion_score > 0 else (0, 255, 0))
            ]

            for index, (label, color) in enumerate(labels):
                cv2.putText(
                    image,
                    label,
                    (20, 30 + index * 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    color,
                    2
                )

        except Exception as error:
            self.last_error = str(error)
            cv2.putText(
                image,
                "Detection error - check dashboard",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 0, 255),
                2
            )

        return av.VideoFrame.from_ndarray(
            image,
            format="bgr24"
        )


st.title("🎓 OA Cheating Detector")
st.caption(
    "AI-assisted online examination monitoring and behavior analytics"
)

st.info(
    "Camera access is requested by your browser. "
    "Detections are indicators for review, not proof of cheating."
)

webrtc_ctx = webrtc_streamer(
    key="oa-monitor",
    video_processor_factory=DetectionProcessor,
    media_stream_constraints={
        "video": True,
        "audio": False
    },
    async_processing=True
)


@st.fragment(run_every="1s")
def show_dashboard_metrics(ctx):

    processor = ctx.video_processor

    if processor is None:
        st.warning(
            "Start the webcam using the control above to see live metrics."
        )
        return

    with processor.lock:
        score = processor.suspicion_score
        status = processor.status

        face_count = processor.face_count
        gaze = processor.gaze_direction

        looking_away = processor.looking_away_active
        phone = processor.phone_detected
        multiple_faces = processor.multiple_faces_active

        ml_looking = processor.ml_looking_away
        ml_phone = processor.ml_phone
        ml_multiple = processor.ml_multiple_faces
        ml_probability = processor.ml_multiple_faces_probability

        error = processor.last_error

        summary = processor.event_logger.get_summary()
        event_rows = list(processor.event_logger.events)

    st.subheader("Live Monitoring")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Suspicion Score", score)
    col2.metric("Status", status)
    col3.metric("Faces Detected", face_count)
    col4.metric("Gaze Direction", gaze)

    st.subheader("Detection Indicators")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Looking Away",
        "Detected" if looking_away else "Normal"
    )

    col2.metric(
        "Phone",
        "Detected" if phone else "Not detected"
    )

    col3.metric(
        "Multiple Faces",
        "Detected" if multiple_faces else "Not detected"
    )

    st.subheader("Session Event Counts")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Events", summary["total"])
    col2.metric("Looking Away", summary["looking_away"])
    col3.metric("Phone", summary["phone"])
    col4.metric("Multiple Faces", summary["multiple_faces"])

    with st.expander("ML Model Predictions"):
        st.write("ML Looking Away:", ml_looking)
        st.write("ML Phone:", ml_phone)
        st.write("ML Multiple Faces:", ml_multiple)
        st.write(
            "Multiple Faces Probability:",
            f"{ml_probability:.2f}"
        )

    if error:
        st.error(f"Detection error: {error}")

    st.subheader("Current Session Event Log")

    if event_rows:
        st.dataframe(
            pd.DataFrame(event_rows),
            use_container_width=True,
            hide_index=True
        )
    else:
        st.write("No suspicious events recorded in this session.")

    event_file = DATASET_DIR / "suspicious_events.csv"
    report_file = DATASET_DIR / "exam_reports.csv"

    if st.button("Save Final Session Report"):
        with processor.lock:
            processor.event_logger.save_report(
                processor.suspicion_score,
                processor.status
            )
            processor.report_saved = True

        st.success("Session report saved.")

    if event_file.exists():
        st.download_button(
            "Download Event Log CSV",
            data=event_file.read_bytes(),
            file_name="suspicious_events.csv",
            mime="text/csv"
        )

    if report_file.exists():
        st.download_button(
            "Download Exam Reports CSV",
            data=report_file.read_bytes(),
            file_name="exam_reports.csv",
            mime="text/csv"
        )


show_dashboard_metrics(webrtc_ctx)

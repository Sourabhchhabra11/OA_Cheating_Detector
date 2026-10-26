from ultralytics import YOLO


class PhoneDetector:

    def __init__(self):

        self.model = YOLO("yolo11n.pt")

    def detect(self, frame):

        results = self.model(
            frame,
            verbose=False
        )

        phones = []

        for result in results:

            for box in result.boxes:

                class_id = int(box.cls[0])

                class_name = self.model.names[class_id]

                confidence = float(box.conf[0])

                if class_name == "cell phone" and confidence >= 0.40:

                    x1, y1, x2, y2 = map(
                        int,
                        box.xyxy[0]
                    )

                    phones.append(
                        {
                            "confidence": confidence,
                            "box": (x1, y1, x2, y2)
                        }
                    )

        return phones
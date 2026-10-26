
import csv
import os
from datetime import datetime


class EventLogger:

    def __init__(
        self,
        event_file="dataset/suspicious_events.csv",
        report_file="dataset/exam_reports.csv"
    ):

        self.event_file = event_file
        self.report_file = report_file
        self.events = []

        os.makedirs(os.path.dirname(event_file), exist_ok=True)
        os.makedirs(os.path.dirname(report_file), exist_ok=True)

        if not os.path.exists(event_file):
            with open(event_file, "w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow([
                    "timestamp",
                    "event",
                    "suspicion_score"
                ])

        if not os.path.exists(report_file):
            with open(report_file, "w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow([
                    "session_time",
                    "total_events",
                    "looking_away",
                    "phone",
                    "multiple_faces",
                    "final_score",
                    "status"
                ])

    def log_event(self, event, suspicion_score):

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        event_data = {
            "timestamp": timestamp,
            "event": event,
            "score": suspicion_score
        }

        self.events.append(event_data)

        with open(self.event_file, "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([
                timestamp,
                event,
                suspicion_score
            ])

        print(
            f"[EVENT] {timestamp} | "
            f"{event} | Score: {suspicion_score}"
        )

    def get_summary(self):

        looking_away = 0
        phone = 0
        multiple_faces = 0

        for event in self.events:
            if event["event"] == "looking_away":
                looking_away += 1
            elif event["event"] == "phone":
                phone += 1
            elif event["event"] == "multiple_faces":
                multiple_faces += 1

        return {
            "total": len(self.events),
            "looking_away": looking_away,
            "phone": phone,
            "multiple_faces": multiple_faces
        }

    def get_status(self, score):

        if score == 0:
            return "NORMAL"
        elif score <= 2:
            return "SUSPICIOUS"
        else:
            return "HIGHLY SUSPICIOUS"

    def save_report(self, final_score=None, final_status=None):

        summary = self.get_summary()

        if final_score is None:
            final_score = max(
                (event["score"] for event in self.events),
                default=0
            )

        if final_status is None:
            final_status = self.get_status(final_score)

        session_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with open(self.report_file, "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([
                session_time,
                summary["total"],
                summary["looking_away"],
                summary["phone"],
                summary["multiple_faces"],
                final_score,
                final_status
            ])

    def print_summary(self, final_score=None, final_status=None):

        summary = self.get_summary()

        if final_score is None:
            final_score = max(
                (event["score"] for event in self.events),
                default=0
            )

        if final_status is None:
            final_status = self.get_status(final_score)

        print()
        print("======================================")
        print("EXAM SESSION SUMMARY")
        print("======================================")
        print()
        print("Total Suspicious Events:", summary["total"])
        print("Looking Away:", summary["looking_away"])
        print("Phone Detected:", summary["phone"])
        print("Multiple Faces:", summary["multiple_faces"])
        print()
        print("Final Suspicion Score:", final_score)
        print("Final Status:", final_status)
        print()
        print("Event Log:")

        for event in self.events:
            print(event["timestamp"], event["event"])

        print()
        print("======================================")

        self.save_report(final_score, final_status)

        print()
        print("Report saved:")
        print(self.report_file)

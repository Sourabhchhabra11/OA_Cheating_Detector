
import pandas as pd

df = pd.read_csv("dataset/behavior_features.csv")

df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)

df["gap"] = df["timestamp"].diff().dt.total_seconds()
df["session_id"] = (df["gap"] > 2).cumsum()

columns = [
    "phone",
    "phone_present",
    "face_count",
    "yaw",
    "pitch",
    "roll",
    "left_gaze",
    "right_gaze"
]

for session in [21, 22, 31]:
    part = df[df["session_id"] == session]

    print("\n" + "=" * 40)
    print("SESSION:", session)
    print("Samples:", len(part))
    print("Phone label distribution:")
    print(part["phone"].value_counts().to_string())

    print("\nFeature averages:")
    print(part[columns].mean(numeric_only=True).round(3).to_string())

    print("\nPhone detector distribution:")
    print(part["phone_present"].value_counts().to_string())


import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df = pd.read_csv("dataset/behavior_features.csv")

df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)
df["gap"] = df["timestamp"].diff().dt.total_seconds()
df["session_id"] = (df["gap"] > 2).cumsum()

features = [
    "face_count",
    "yaw",
    "pitch",
    "roll",
    "left_gaze",
    "right_gaze",
    "gaze_direction",
    "phone_present"
]

test_sessions = [16, 20, 22, 25, 31]
test = df[df["session_id"].isin(test_sessions)].copy()

y_true = test["phone"].astype(int)
baseline_predictions = test["phone_present"].astype(int)

preprocessor = joblib.load("models/behavior_preprocessor.pkl")
model = joblib.load("models/phone_model.pkl")

X = preprocessor.transform(test[features])
model_predictions = model.predict(X)

for name, predictions in [
    ("Phone detector baseline", baseline_predictions),
    ("Machine learning model", model_predictions)
]:
    print("\n" + "=" * 45)
    print(name)
    print("=" * 45)

    print("Accuracy:", round(accuracy_score(y_true, predictions), 4))
    print("\nConfusion matrix:")
    print(confusion_matrix(y_true, predictions))

    print("\nClassification report:")
    print(classification_report(
        y_true,
        predictions,
        target_names=["Normal", "Phone"],
        zero_division=0
    ))

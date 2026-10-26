import pandas as pd
import joblib

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


df = pd.read_csv("dataset/behavior_features.csv")

df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)

df["gap"] = df["timestamp"].diff().dt.total_seconds()
df["session_id"] = (df["gap"] > 2).cumsum()


test_sessions = [16, 20, 22, 25, 31]

test = df[
    df["session_id"].isin(test_sessions)
].copy()


# Direct multiple-face detector

test["direct_prediction"] = (
    test["face_count"] > 1
).astype(int)


# ML model

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

preprocessor = joblib.load(
    "models/behavior_preprocessor.pkl"
)

model = joblib.load(
    "models/multiple_faces_model.pkl"
)

X = preprocessor.transform(
    test[features]
)

ml_prediction = model.predict(X)

y_true = test["multiple_faces"].astype(int)


print()
print("=" * 45)
print("DIRECT MULTIPLE-FACES DETECTOR")
print("=" * 45)

print(
    "Accuracy:",
    round(
        accuracy_score(
            y_true,
            test["direct_prediction"]
        ),
        4
    )
)

print()
print("Confusion Matrix:")
print(
    confusion_matrix(
        y_true,
        test["direct_prediction"]
    )
)

print()
print("Classification Report:")
print(
    classification_report(
        y_true,
        test["direct_prediction"],
        target_names=[
            "Normal",
            "Multiple Faces"
        ],
        zero_division=0
    )
)


print()
print("=" * 45)
print("ML MULTIPLE-FACES MODEL")
print("=" * 45)

print(
    "Accuracy:",
    round(
        accuracy_score(
            y_true,
            ml_prediction
        ),
        4
    )
)

print()
print("Confusion Matrix:")
print(
    confusion_matrix(
        y_true,
        ml_prediction
    )
)

print()
print("Classification Report:")
print(
    classification_report(
        y_true,
        ml_prediction,
        target_names=[
            "Normal",
            "Multiple Faces"
        ],
        zero_division=0
    )
)
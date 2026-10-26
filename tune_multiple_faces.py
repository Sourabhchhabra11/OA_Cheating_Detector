import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


df = pd.read_csv(
    "dataset/behavior_features.csv"
)

df["timestamp"] = pd.to_datetime(
    df["timestamp"]
)

df = df.sort_values(
    "timestamp"
).reset_index(drop=True)

df["gap"] = (
    df["timestamp"].diff().dt.total_seconds()
)

df["session_id"] = (
    df["gap"] > 2
).cumsum()


test_sessions = [
    16,
    20,
    22,
    25,
    31
]


test = df[
    df["session_id"].isin(test_sessions)
].copy()


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

y_true = test[
    "multiple_faces"
].astype(int)


probabilities = model.predict_proba(X)[:, 1]


print()
print("=" * 60)
print("MULTIPLE-FACES THRESHOLD TUNING")
print("=" * 60)

print()

best_threshold = None
best_f1 = 0


thresholds = [
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80
]


print(
    "Threshold | Accuracy | Precision | Recall | F1"
)

print("-" * 60)


for threshold in thresholds:

    predictions = (
        probabilities >= threshold
    ).astype(int)

    accuracy = accuracy_score(
        y_true,
        predictions
    )

    precision = precision_score(
        y_true,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        predictions,
        zero_division=0
    )

    print(
        f"{threshold:9.2f} | "
        f"{accuracy:8.4f} | "
        f"{precision:9.4f} | "
        f"{recall:6.4f} | "
        f"{f1:6.4f}"
    )

    if f1 > best_f1:

        best_f1 = f1
        best_threshold = threshold


print()

print("=" * 60)
print("BEST THRESHOLD")
print("=" * 60)

print(
    "Threshold:",
    best_threshold
)

print(
    "F1:",
    round(best_f1, 4)
)
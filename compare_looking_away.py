import pandas as pd
import joblib

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


df = pd.read_csv("dataset/behavior_features.csv")

df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)

df["gap"] = df["timestamp"].diff().dt.total_seconds()
df["session_id"] = (df["gap"] > 2).cumsum()


test_sessions = [16, 20, 22, 25, 31]

test = df[df["session_id"].isin(test_sessions)].copy()


# Direct detector
#
# Looking away is considered suspicious when:
# 1. Head is turned beyond the threshold
# OR
# 2. Gaze stays LEFT/RIGHT
#
# We are evaluating the raw behavior here,
# without the 1.5 second temporal tracker.

HEAD_THRESHOLD = 15

test["head_turned"] = (
    test["yaw"].abs() > HEAD_THRESHOLD
)

test["gaze_away"] = (
    test["gaze_direction"] != "CENTER"
)

test["direct_prediction"] = (
    test["head_turned"] |
    test["gaze_away"]
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
    "models/looking_away_model.pkl"
)

X = preprocessor.transform(
    test[features]
)

ml_prediction = model.predict(X)


y_true = test["looking_away"].astype(int)


print()
print("=" * 45)
print("DIRECT LOOKING-AWAY DETECTOR")
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
            "Looking Away"
        ],
        zero_division=0
    )
)


print()
print("=" * 45)
print("ML LOOKING-AWAY MODEL")
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
            "Looking Away"
        ],
        zero_division=0
    )
)
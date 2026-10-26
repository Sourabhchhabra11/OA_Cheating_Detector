import pandas as pd
import joblib
import os

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
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
    df["timestamp"]
    .diff()
    .dt.total_seconds()
)


df["session_id"] = (
    df["gap"] > 2
).cumsum()


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


test_sessions = {
    "normal": 16,
    "looking_away": 20,
    "phone": 22,
    "multiple_faces": 25,
    "combined": 31
}


test_session_ids = list(
    test_sessions.values()
)


test_mask = df["session_id"].isin(
    test_session_ids
)


test_df = df[test_mask].copy()


X_test = test_df[features]


preprocessor = joblib.load(
    "models/behavior_preprocessor.pkl"
)


X_test = preprocessor.transform(
    X_test
)


models = {
    "looking_away":
        joblib.load(
            "models/looking_away_model.pkl"
        ),

    "phone":
        joblib.load(
            "models/phone_model.pkl"
        ),

    "multiple_faces":
        joblib.load(
            "models/multiple_faces_model.pkl"
        )
}


print()
print("======================================")
print("FINAL MODEL EVALUATION")
print("======================================")

print()

print(
    "Testing samples:",
    len(test_df)
)

print()

results = {}


for target, model in models.items():

    print()
    print("======================================")
    print(
        f"{target.upper()} MODEL"
    )
    print("======================================")

    y_test = test_df[target]

    predictions = model.predict(
        X_test
    )


    accuracy = accuracy_score(
        y_test,
        predictions
    )


    results[target] = accuracy


    print()

    print(
        f"Accuracy: {accuracy:.4f}"
    )

    print()

    print(
        classification_report(
            y_test,
            predictions,
            target_names=[
                "Normal",
                "Suspicious"
            ],
            zero_division=0
        )
    )


    print(
        "Confusion Matrix:"
    )

    print(
        confusion_matrix(
            y_test,
            predictions
        )
    )


print()
print("======================================")
print("FINAL ACCURACY COMPARISON")
print("======================================")

print()

for target, accuracy in results.items():

    print(
        f"{target}: {accuracy:.4f}"
    )


average_accuracy = (
    sum(results.values())
    / len(results)
)


print()

print(
    f"Average Behavior Accuracy: "
    f"{average_accuracy:.4f}"
)


print()
print("======================================")
print("EVALUATION COMPLETE")
print("======================================")
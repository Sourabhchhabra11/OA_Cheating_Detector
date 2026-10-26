import os
import joblib
import pandas as pd

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


df = pd.read_csv("dataset/behavior_features.csv")

df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)

df["gap"] = df["timestamp"].diff().dt.total_seconds()
df["session_id"] = (df["gap"] > 2).cumsum()

test_session_ids = [16, 20, 22, 25, 31]

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

targets = [
    "looking_away",
    "phone",
    "multiple_faces"
]

test_df = df[
    df["session_id"].isin(test_session_ids)
].copy()

X_test = test_df[features]

preprocessor = joblib.load(
    "models/behavior_preprocessor.pkl"
)

X_test_transformed = preprocessor.transform(X_test)

os.makedirs("reports/error_analysis", exist_ok=True)

all_results = []

print("=" * 50)
print("MODEL ERROR ANALYSIS")
print("=" * 50)
print("Test samples:", len(test_df))
print("Test sessions:", test_session_ids)


for target in targets:

    print("\n" + "=" * 50)
    print(target.upper())
    print("=" * 50)

    model = joblib.load(
        f"models/{target}_model.pkl"
    )

    actual = test_df[target].astype(int)
    predicted = model.predict(X_test_transformed)

    print(f"\nAccuracy: {accuracy_score(actual, predicted):.4f}")

    print("\nConfusion matrix:")
    print("Rows = actual, columns = predicted")
    print(confusion_matrix(actual, predicted, labels=[0, 1]))

    print("\nClassification report:")
    print(
        classification_report(
            actual,
            predicted,
            labels=[0, 1],
            target_names=["Normal", "Suspicious"],
            zero_division=0
        )
    )

    errors = test_df.copy()

    errors["actual"] = actual
    errors["predicted"] = predicted

    errors["result"] = "Correct"

    errors.loc[
        (actual == 1) & (predicted == 0),
        "result"
    ] = "False Negative"

    errors.loc[
        (actual == 0) & (predicted == 1),
        "result"
    ] = "False Positive"

    errors.loc[
        (actual == 1) & (predicted == 1),
        "result"
    ] = "True Positive"

    errors.loc[
        (actual == 0) & (predicted == 0),
        "result"
    ] = "True Negative"

    errors["behavior"] = target

    false_negatives = errors[
        errors["result"] == "False Negative"
    ]

    false_positives = errors[
        errors["result"] == "False Positive"
    ]

    print("False negatives:", len(false_negatives))
    print("False positives:", len(false_positives))

    output_columns = [
        "timestamp",
        "session_id",
        "behavior",
        "actual",
        "predicted",
        "result"
    ] + features

    errors[output_columns].to_csv(
        f"reports/error_analysis/{target}_errors.csv",
        index=False
    )

    numeric_features = [
        "face_count",
        "yaw",
        "pitch",
        "roll",
        "left_gaze",
        "right_gaze",
        "phone_present"
    ]

    groups = errors.groupby("result")[numeric_features].mean().round(3)

    print("\nAverage numerical features by result:")
    print(groups.to_string())

    print("\nGaze direction counts by result:")
    print(
        pd.crosstab(
            errors["result"],
            errors["gaze_direction"]
        ).to_string()
    )

    summary = {
        "behavior": target,
        "accuracy": accuracy_score(actual, predicted),
        "false_negatives": len(false_negatives),
        "false_positives": len(false_positives),
        "true_positives": int(
            ((actual == 1) & (predicted == 1)).sum()
        ),
        "true_negatives": int(
            ((actual == 0) & (predicted == 0)).sum()
        )
    }

    all_results.append(summary)


summary_df = pd.DataFrame(all_results)

summary_df.to_csv(
    "reports/error_analysis/summary.csv",
    index=False
)

print("\n" + "=" * 50)
print("ERROR ANALYSIS COMPLETE")
print("=" * 50)

print(summary_df.to_string(index=False))

print("\nFiles saved in reports/error_analysis/")
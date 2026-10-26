import pandas as pd
import os
import joblib

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


df = pd.read_csv(
    "dataset/behavior_features.csv"
)


df["timestamp"] = pd.to_datetime(
    df["timestamp"]
)

df = df.sort_values(
    "timestamp"
).reset_index(drop=True)


df["gap"] = df["timestamp"].diff().dt.total_seconds()

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


categorical_features = [
    "gaze_direction"
]


numerical_features = [
    "face_count",
    "yaw",
    "pitch",
    "roll",
    "left_gaze",
    "right_gaze",
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


train_df = df[~test_mask].copy()

test_df = df[test_mask].copy()


X_train = train_df[features]

X_test = test_df[features]


preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            categorical_features
        ),
        (
            "numerical",
            "passthrough",
            numerical_features
        )
    ]
)


X_train = preprocessor.fit_transform(
    X_train
)

X_test = preprocessor.transform(
    X_test
)


print("======================================")
print("BEHAVIOR MODEL TRAINING")
print("======================================")

print()

print("Training samples:", len(train_df))
print("Testing samples:", len(test_df))

print()

print("Training sessions:")
print(sorted(train_df["session_id"].unique()))

print()

print("Testing sessions:")
print(sorted(test_df["session_id"].unique()))


targets = [
    "looking_away",
    "phone",
    "multiple_faces"
]


models = {}


for target in targets:

    print()
    print("======================================")
    print(f"TRAINING: {target.upper()}")
    print("======================================")

    y_train = train_df[target]

    y_test = test_df[target]

    print()

    print("Training distribution:")
    print(y_train.value_counts())

    print()

    print("Testing distribution:")
    print(y_test.value_counts())

    model = RandomForestClassifier(
        n_estimators=150,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    )

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    print()

    print(
        f"Accuracy: {accuracy:.4f}"
    )

    print()

    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0
        )
    )

    models[target] = model


os.makedirs(
    "models",
    exist_ok=True
)


for target, model in models.items():

    joblib.dump(
        model,
        f"models/{target}_model.pkl"
    )


joblib.dump(
    preprocessor,
    "models/behavior_preprocessor.pkl"
)


print()
print("======================================")
print("ALL BEHAVIOR MODELS SAVED")
print("======================================")

print()

print("models/looking_away_model.pkl")
print("models/phone_model.pkl")
print("models/multiple_faces_model.pkl")
print("models/behavior_preprocessor.pkl")
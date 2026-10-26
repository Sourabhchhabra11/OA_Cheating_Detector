
import pandas as pd
import joblib

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

preprocessor = joblib.load("models/behavior_preprocessor.pkl")
model = joblib.load("models/phone_model.pkl")

test_sessions = [16, 20, 22, 25, 31]
test = df[df["session_id"].isin(test_sessions)].copy()

X = preprocessor.transform(test[features])

test["predicted"] = model.predict(X)
test["phone_probability"] = model.predict_proba(X)[:, 1]

print("\nFEATURE IMPORTANCE")
print("=" * 50)

feature_names = preprocessor.get_feature_names_out()

importance = pd.DataFrame({
    "feature": feature_names,
    "importance": model.feature_importances_
})

print(
    importance.sort_values("importance", ascending=False)
    .to_string(index=False)
)

for session in [22, 31]:
    part = test[test["session_id"] == session]

    print("\n" + "=" * 50)
    print("SESSION:", session)

    print("\nPredictions by phone detector:")
    print(pd.crosstab(
        part["phone_present"],
        part["predicted"],
        rownames=["phone_present"],
        colnames=["predicted"]
    ))

    print("\nPhone prediction probability:")
    print(
        part.groupby("phone_present")["phone_probability"]
        .agg(["count", "mean", "min", "max"])
        .round(3)
        .to_string()
    )

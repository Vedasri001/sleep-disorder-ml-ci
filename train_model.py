import json
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def train_model():
    print("Loading sleep dataset...")
    data = pd.read_csv("sleep_dataset.csv")

    # Assumption: blank target values mean No Disorder.
    # Confirm this interpretation with your instructor.
    data["Sleep Disorder"] = data["Sleep Disorder"].fillna("No Disorder")
    data["Sleep Disorder"] = data["Sleep Disorder"].astype(str).str.strip()
    data.loc[data["Sleep Disorder"] == "", "Sleep Disorder"] = "No Disorder"

    # Convert blood pressure from text such as 125/88 into numbers.
    bp = data["Blood Pressure"].astype(str).str.extract(
        r"^\s*(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)\s*$"
    )
    data["Systolic BP"] = pd.to_numeric(bp[0], errors="coerce")
    data["Diastolic BP"] = pd.to_numeric(bp[1], errors="coerce")

    # Remove identifier and original text-form blood pressure.
    data = data.drop(columns=["Person ID", "Blood Pressure"])
    data.to_csv("sleep_dataset_processed.csv", index=False)

    print("Records:", len(data))
    print("Target distribution:")
    print(data["Sleep Disorder"].value_counts())

    X = data.drop(columns=["Sleep Disorder"])
    y = data["Sleep Disorder"]

    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=["number"]).columns.tolist()

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessing = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features)
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    print("Training records:", len(X_train))
    print("Testing records:", len(X_test))

    model = Pipeline([
        ("preprocessing", preprocessing),
        ("classifier", LogisticRegression(max_iter=2000, random_state=42))
    ])

    print("Training model...")
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    labels = sorted(y.unique().tolist())
    matrix = confusion_matrix(y_test, predictions, labels=labels)

    print("\nModel Evaluation")
    print("----------------")
    print("Accuracy:", round(float(accuracy), 4))
    print("Class labels:", labels)
    print("Confusion Matrix:")
    print(matrix)
    print("\nClassification Report:")
    print(classification_report(y_test, predictions, zero_division=0))

    joblib.dump({
        "model": model,
        "features": X.columns.tolist(),
        "labels": labels
    }, "sleep_disorder_model.pkl")

    metrics = {
        "accuracy": float(accuracy),
        "training_records": len(X_train),
        "testing_records": len(X_test),
        "total_records": len(data),
        "target_classes": labels,
        "confusion_matrix": matrix.tolist()
    }

    with open("metrics.json", "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    print("Model saved as sleep_disorder_model.pkl")
    print("Metrics saved as metrics.json")


if __name__ == "__main__":
    train_model()

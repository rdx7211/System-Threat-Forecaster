import os
import joblib
import pandas as pd

# ------------------------------
# File handling utils
# ------------------------------
def save_model(model, path="models/final_pipeline.pkl"):
    """Save trained model/pipeline to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    print(f"✅ Model saved at {path}")


def load_model(path="models/final_pipeline.pkl"):
    """Load trained model/pipeline from disk."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ Model file not found: {path}")
    model = joblib.load(path)
    print(f"✅ Model loaded from {path}")
    return model


# ------------------------------
# Data utils
# ------------------------------
def load_data(train_path, test_path=None):
    """Load train (and optionally test) data from CSV."""
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path) if test_path else None
    return train_df, test_df


def save_submission(ids, predictions, path="submission.csv"):
    """Save predictions into a submission file (Kaggle-style)."""
    submission = pd.DataFrame({
        "id": ids,
        "target": predictions
    })
    submission.to_csv(path, index=False)
    print(f"✅ Submission file saved at {path}")

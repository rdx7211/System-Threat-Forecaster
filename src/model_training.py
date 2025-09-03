import os
import joblib
import lightgbm as lgb
from sklearn.pipeline import Pipeline

def train_and_save_model(preprocessor, X, y, model_path="models/final_pipeline.pkl"):
    """
    Train the best LightGBM model on the given data and save the pipeline.

    Args:
        preprocessor: sklearn ColumnTransformer (handles preprocessing)
        X (DataFrame): Training features
        y (Series): Target labels
        model_path (str): Path to save trained pipeline

    Returns:
        pipeline: trained sklearn Pipeline
    """

    # ✅ Define best LightGBM model inside pipeline
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", lgb.LGBMClassifier(
            n_estimators=1200,
            learning_rate=0.01,
            max_depth=14,
            num_leaves=80,
            min_child_samples=50,
            colsample_bytree=0.8,
            subsample=0.8,
            reg_alpha=0.5,
            reg_lambda=0.7,
            min_split_gain=0.05,
            boosting_type="gbdt",
            random_state=42,
            verbose=-1
        ))
    ])

    # ✅ Train on full dataset
    pipeline.fit(X, y)

    # ✅ Save trained pipeline
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(pipeline, model_path)
    print(f"✅ Model saved at {model_path}")

    return pipeline

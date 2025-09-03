import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, roc_auc_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)

def evaluate_model(model, X_train, y_train, X_val, y_val):
    """
    Evaluate model performance on training and validation sets.
    Prints metrics and shows confusion matrices.

    Args:
        model: trained sklearn Pipeline or estimator
        X_train, y_train: training data
        X_val, y_val: validation data
    """

    # Training predictions
    y_train_pred = model.predict(X_train)
    y_train_proba = model.predict_proba(X_train)[:, 1]

    # Validation predictions
    y_val_pred = model.predict(X_val)
    y_val_proba = model.predict_proba(X_val)[:, 1]

    # -------------------------------
    # Training Metrics
    # -------------------------------
    print("\n📊 Training Metrics:")
    print(f"Accuracy: {accuracy_score(y_train, y_train_pred):.4f}")
    print(f"AUC: {roc_auc_score(y_train, y_train_proba):.4f}")
    print("Classification Report:")
    print(classification_report(y_train, y_train_pred))

    cm_train = confusion_matrix(y_train, y_train_pred)
    disp_train = ConfusionMatrixDisplay(confusion_matrix=cm_train)
    disp_train.plot(cmap="Blues")
    plt.title("Training Confusion Matrix")
    plt.show()

    # -------------------------------
    # Validation Metrics
    # -------------------------------
    print("\n📊 Validation Metrics:")
    print(f"Accuracy: {accuracy_score(y_val, y_val_pred):.4f}")
    print(f"AUC: {roc_auc_score(y_val, y_val_proba):.4f}")
    print("Classification Report:")
    print(classification_report(y_val, y_val_pred))

    cm_val = confusion_matrix(y_val, y_val_pred)
    disp_val = ConfusionMatrixDisplay(confusion_matrix=cm_val)
    disp_val.plot(cmap="Greens")
    plt.title("Validation Confusion Matrix")
    plt.show()

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer

def preprocess_raw_data(train_df, test_df):
    """
    Prepares raw train and test DataFrames:
    - Drops MachineID
    - Handles date features
    - Drops columns with >50% missing values
    - Drops duplicates (train only)
    """

    # Drop MachineID
    if "MachineID" in train_df.columns:
        train_df = train_df.drop(columns=["MachineID"])
    if "MachineID" in test_df.columns:
        test_df = test_df.drop(columns=["MachineID"])

    # Handle date columns
    for df in [train_df, test_df]:
        df["DateAS"] = pd.to_datetime(df["DateAS"], errors="coerce")
        df["DateOS"] = pd.to_datetime(df["DateOS"], errors="coerce")

        df["AS_Year"] = df["DateAS"].dt.year
        df["AS_Month"] = df["DateAS"].dt.month
        df["OS_Year"] = df["DateOS"].dt.year
        df["OS_Month"] = df["DateOS"].dt.month

        df.drop(columns=["DateAS", "DateOS"], inplace=True)

    # Drop columns with >50% missing values
    missing_threshold = 50
    missing_percentage = (train_df.isnull().sum() / len(train_df)) * 100
    cols_to_drop = missing_percentage[missing_percentage > missing_threshold].index.tolist()

    train_df = train_df.drop(columns=cols_to_drop)
    test_df = test_df.drop(columns=cols_to_drop)

    # Drop duplicates
    train_df = train_df.drop_duplicates()

    return train_df, test_df


def create_preprocessor(X: pd.DataFrame):
    """
    Creates preprocessing pipeline for numeric + categorical columns
    """

    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()

    # Pipelines
    num_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    cat_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
    ])

    preprocessor = ColumnTransformer([
        ("num", num_transformer, num_cols),
        ("cat", cat_transformer, cat_cols)
    ])

    return preprocessor, num_cols, cat_cols

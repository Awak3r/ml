import pandas as pd


def drop_duplicates(df):
    before = len(df)
    df = df.drop_duplicates()
    print("Удалено дубликатов:", before - len(df))
    return df


def fill_missing(df):
    df = df.copy()
    num_cols = df.select_dtypes(include=["float64", "int64"]).columns
    for col in num_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())
    cat_cols = df.select_dtypes(include=["object", "string"]).columns
    for col in cat_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mode()[0])
    print("Пропусков осталось:", df.isnull().sum().sum())
    return df


def filter_risk(df, lo=0, hi=100):
    before = len(df)
    df = df[(df["RiskScore"] >= lo) & (df["RiskScore"] <= hi)]
    print("Удалено выбросов RiskScore:", before - len(df))
    return df


def prepare_features(train_df, test_df, target="RiskScore"):
    X = train_df.drop(columns=["ApplicationDate", target])
    y = train_df[target].values
    X_test = test_df.drop(columns=["ApplicationDate", "ID"])
    cat_cols = X.select_dtypes(include=["object", "string"]).columns
    X = pd.get_dummies(X, columns=cat_cols, drop_first=True).astype(float)
    X_test = pd.get_dummies(X_test, columns=cat_cols, drop_first=True).astype(float)
    X_test = X_test.reindex(columns=X.columns, fill_value=0)
    return X, y, X_test

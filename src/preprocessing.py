"""
preprocessing.py - Data Preprocessing & Feature Engineering Pipeline.

Provides scikit-learn compatible custom transformers and pipeline constructors
for the Loan Default Prediction System.
"""

from typing import List, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Canonical column definitions
CATEGORICAL_FEATURES: List[str] = [
    "Gender",
    "Married",
    "Dependents",
    "Education",
    "Self_Employed",
    "Credit_History",
    "Property_Area",
]

NUMERIC_BASE_FEATURES: List[str] = [
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term",
]

ENGINEERED_NUMERIC_FEATURES: List[str] = [
    "Total_Income",
    "Total_Income_Log",
    "LoanAmount_Log",
    "EMI",
    "Balance_Income",
]

ALL_NUMERIC_FEATURES: List[str] = NUMERIC_BASE_FEATURES + ENGINEERED_NUMERIC_FEATURES


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Custom Scikit-Learn Transformer for domain-specific credit risk feature engineering.
    
    Creates:
    - Total_Income = ApplicantIncome + CoapplicantIncome
    - Total_Income_Log = log(1 + Total_Income)
    - LoanAmount_Log = log(1 + LoanAmount)
    - EMI = (LoanAmount * 1000) / Loan_Amount_Term
    - Balance_Income = Total_Income - EMI
    """

    def __init__(self, drop_identifiers: bool = True):
        self.drop_identifiers = drop_identifiers

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: np.ndarray = None):
        return self

    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        
        # Work on a copy to prevent SettingWithCopy warnings
        df = X.copy()

        # Remove Loan_ID if present
        if self.drop_identifiers and "Loan_ID" in df.columns:
            df = df.drop(columns=["Loan_ID"])

        # Ensure base numeric columns exist and are numeric
        for col in NUMERIC_BASE_FEATURES:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # 1. Total Income & Log Transform
        app_inc = df["ApplicantIncome"].fillna(0) if "ApplicantIncome" in df.columns else 0
        coapp_inc = df["CoapplicantIncome"].fillna(0) if "CoapplicantIncome" in df.columns else 0
        total_income = app_inc + coapp_inc
        df["Total_Income"] = total_income
        df["Total_Income_Log"] = np.log1p(np.maximum(0, total_income))

        # 2. Loan Amount Log Transform (preserving NaNs for pipeline imputation)
        if "LoanAmount" in df.columns:
            df["LoanAmount_Log"] = np.where(
                df["LoanAmount"].notnull(),
                np.log1p(np.maximum(0, df["LoanAmount"])),
                np.nan,
            )
        else:
            df["LoanAmount_Log"] = np.nan

        # 3. Equated Monthly Installment (EMI)
        # Note: LoanAmount is in thousands ($K), Loan_Amount_Term is in months.
        if "LoanAmount" in df.columns and "Loan_Amount_Term" in df.columns:
            term_safe = np.where(df["Loan_Amount_Term"] > 0, df["Loan_Amount_Term"], 360.0)
            emi = (df["LoanAmount"] * 1000.0) / term_safe
            df["EMI"] = np.where(df["LoanAmount"].notnull() & df["Loan_Amount_Term"].notnull(), emi, np.nan)
        else:
            df["EMI"] = np.nan

        # 4. Balance Income (Discretionary Income after loan installment)
        if "EMI" in df.columns:
            df["Balance_Income"] = np.where(
                df["EMI"].notnull(),
                df["Total_Income"] - df["EMI"],
                np.nan,
            )
        else:
            df["Balance_Income"] = np.nan

        # Ensure categorical features are strings for imputer and one-hot encoder
        for col in CATEGORICAL_FEATURES:
            if col in df.columns:
                df[col] = df[col].astype(object)
                # Convert numeric floats like 1.0 or 0.0 to uniform strings '1.0' / '0.0' while preserving NaN
                df[col] = df[col].apply(lambda v: str(v) if pd.notnull(v) else np.nan)

        return df


# Class alias for notebook compatibility
CreditRiskFeatureEngineer = FeatureEngineer


def build_column_preprocessor() -> ColumnTransformer:
    """
    Constructs the ColumnTransformer for numeric and categorical pipelines.
    
    Numeric Strategy:
    - Median Imputation (robust against skewness & outliers)
    - StandardScaler (zero mean, unit variance)
    
    Categorical Strategy:
    - Most Frequent (Mode) Imputation
    - OneHotEncoder (drop='first' to prevent collinearity, handle_unknown='ignore')
    """
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, ALL_NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )

    return preprocessor


def build_full_pipeline(classifier: BaseEstimator) -> Pipeline:
    """
    Constructs the complete end-to-end inference pipeline:
    Feature Engineering -> Imputation & Encoding -> Model Estimation.
    
    Parameters:
    - classifier: Scikit-learn classification estimator (e.g., LogisticRegression, RandomForestClassifier)
    
    Returns:
    - Complete Scikit-learn Pipeline
    """
    return Pipeline(
        steps=[
            ("feature_engineer", FeatureEngineer(drop_identifiers=True)),
            ("preprocessor", build_column_preprocessor()),
            ("classifier", classifier),
        ]
    )


def load_and_clean_data(file_path: str) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Loads raw CSV data, separates target 'Loan_Status', drops identifier columns,
    and maps target labels to binary values (1: Approved, 0: Default/Rejected).
    
    Parameters:
    - file_path: Path to loan_data.csv
    
    Returns:
    - X: Features DataFrame
    - y: Target Series (1 for 'Y', 0 for 'N')
    """
    df = pd.read_csv(file_path)
    
    if "Loan_ID" in df.columns:
        df = df.drop(columns=["Loan_ID"])

    if "Loan_Status" not in df.columns:
        raise ValueError("Target column 'Loan_Status' not found in dataset.")

    # Target Mapping: 'Y' -> 1 (Loan Approved / No Default), 'N' -> 0 (Rejected / Default Risk)
    target_map = {"Y": 1, "N": 0}
    y = df["Loan_Status"].map(target_map)
    
    if y.isnull().any():
        raise ValueError("Invalid target values detected in 'Loan_Status'. Expected 'Y' or 'N'.")

    X = df.drop(columns=["Loan_Status"])
    return X, y

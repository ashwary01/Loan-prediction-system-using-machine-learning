"""
predict.py - Model Inference Utility for Loan Default Prediction.

Loads the serialized pipeline artifact and exposes prediction functions
for single applicant records, batches, and CLI JSON payloads.

Usage:
    python src/predict.py
    python src/predict.py --applicant '{"Gender": "Male", "Married": "Yes", "ApplicantIncome": 5000, ...}'
"""

import argparse
import json
import os
import sys
from typing import Dict, Any, List, Union
import numpy as np
import pandas as pd
import joblib

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.preprocessing import FeatureEngineer

# Ensure compatibility if artifact was serialized from notebook __main__
sys.modules["__main__"].CreditRiskFeatureEngineer = FeatureEngineer

# Required input features
EXPECTED_COLUMNS = [
    "Gender",
    "Married",
    "Dependents",
    "Education",
    "Self_Employed",
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term",
    "Credit_History",
    "Property_Area",
]


class LoanPredictor:
    """
    Inference engine that wraps the trained Pipeline artifact.
    """

    def __init__(self, model_path: str = "models/loan_model.joblib"):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model artifact not found at '{model_path}'. "
                "Please run 'python src/train.py' first to train and export the model."
            )
        self.model_path = model_path
        self.pipeline = joblib.load(model_path)

    def _format_input(self, data: Union[Dict[str, Any], List[Dict[str, Any]], pd.DataFrame]) -> pd.DataFrame:
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        elif isinstance(data, list):
            df = pd.DataFrame(data)
        elif isinstance(data, pd.DataFrame):
            df = data.copy()
        else:
            raise TypeError("Input data must be a dictionary, list of dictionaries, or pandas DataFrame.")

        # Ensure all expected columns exist; fill missing columns with NaN
        for col in EXPECTED_COLUMNS:
            if col not in df.columns:
                df[col] = np.nan

        return df[EXPECTED_COLUMNS]

    def predict(self, applicant_data: Union[Dict[str, Any], List[Dict[str, Any]], pd.DataFrame]) -> List[Dict[str, Any]]:
        """
        Generates risk assessments, binary decisions, and default probabilities.
        
        Returns:
            List of prediction dictionaries containing approval status, probabilities,
            and credit risk classification.
        """
        df_input = self._format_input(applicant_data)
        
        preds = self.pipeline.predict(df_input)
        probas = self.pipeline.predict_proba(df_input)  # [P(Class 0: Default), P(Class 1: Approved)]

        results = []
        for i, row in df_input.iterrows():
            pred = int(preds[i])
            p_default = float(probas[i][0])
            p_approved = float(probas[i][1])

            # Classify risk tier based on default probability
            if p_default < 0.25:
                risk_tier = "Low"
            elif p_default < 0.50:
                risk_tier = "Moderate"
            elif p_default < 0.75:
                risk_tier = "High"
            else:
                risk_tier = "Very High"

            # Domain heuristics for risk explanation
            risk_factors = []
            ch = row.get("Credit_History")
            if pd.isna(ch) or str(ch).strip() in ["0", "0.0"]:
                risk_factors.append("Derogatory or missing credit history (Credit_History <= 0)")
            
            app_inc = float(row.get("ApplicantIncome") or 0)
            coapp_inc = float(row.get("CoapplicantIncome") or 0)
            tot_inc = app_inc + coapp_inc
            loan_amt = float(row.get("LoanAmount") or 0) * 1000.0  # in dollars
            
            if tot_inc > 0 and (loan_amt / (tot_inc * 12)) > 0.40:
                risk_factors.append("High debt-to-income ratio (Loan exceeds 40% annual income)")
            if tot_inc < 2500:
                risk_factors.append("Low household total income threshold (< $2,500/mo)")

            result = {
                "Loan_Status": "Y" if pred == 1 else "N",
                "Decision": "Approved" if pred == 1 else "Rejected",
                "Approval_Probability": round(p_approved, 4),
                "Default_Probability": round(p_default, 4),
                "Risk_Tier": risk_tier,
                "Key_Risk_Factors": risk_factors if risk_factors else ["Strong credit profile & financial buffers"],
            }
            results.append(result)

        return results


def main():
    parser = argparse.ArgumentParser(description="Inference tool for Loan Default Prediction")
    parser.add_argument("--model_path", type=str, default="models/loan_model.joblib", help="Path to trained model")
    parser.add_argument("--applicant", type=str, default=None, help="JSON string representing single applicant")
    parser.add_argument("--input_file", type=str, default=None, help="Path to JSON file with applicant data")
    
    # Direct field flags for convenient CLI invocation
    parser.add_argument("--gender", type=str, default=None, help="Male or Female")
    parser.add_argument("--married", type=str, default=None, help="Yes or No")
    parser.add_argument("--dependents", type=str, default=None, help="0, 1, 2, or 3+")
    parser.add_argument("--education", type=str, default=None, help="Graduate or Not Graduate")
    parser.add_argument("--self_employed", type=str, default=None, help="Yes or No")
    parser.add_argument("--applicant_income", type=float, default=None, help="Monthly applicant income ($)")
    parser.add_argument("--coapplicant_income", type=float, default=None, help="Monthly co-applicant income ($)")
    parser.add_argument("--loan_amount", type=float, default=None, help="Loan amount in thousands ($K)")
    parser.add_argument("--loan_term", type=float, default=None, help="Loan term in months (e.g. 360)")
    parser.add_argument("--credit_history", type=float, default=None, help="1.0 (Good) or 0.0 (Poor)")
    parser.add_argument("--property_area", type=str, default=None, help="Urban, Semiurban, or Rural")
    
    args = parser.parse_args()

    predictor = LoanPredictor(model_path=args.model_path)

    # Check if direct field arguments were supplied
    if args.applicant_income is not None or args.loan_amount is not None or args.credit_history is not None:
        payload = {
            "Gender": args.gender or "Male",
            "Married": args.married or "Yes",
            "Dependents": args.dependents or "0",
            "Education": args.education or "Graduate",
            "Self_Employed": args.self_employed or "No",
            "ApplicantIncome": args.applicant_income if args.applicant_income is not None else 5000,
            "CoapplicantIncome": args.coapplicant_income if args.coapplicant_income is not None else 0,
            "LoanAmount": args.loan_amount if args.loan_amount is not None else 120.0,
            "Loan_Amount_Term": args.loan_term if args.loan_term is not None else 360.0,
            "Credit_History": args.credit_history if args.credit_history is not None else 1.0,
            "Property_Area": args.property_area or "Semiurban",
        }
        result = predictor.predict(payload)
        print("\nPrediction Result:")
        print(json.dumps(result[0], indent=2))
        return

    if args.applicant:
        payload = json.loads(args.applicant)
        result = predictor.predict(payload)
        print("\nPrediction Result:")
        print(json.dumps(result, indent=2))
    elif args.input_file:
        with open(args.input_file, "r") as f:
            payload = json.load(f)
        result = predictor.predict(payload)
        print("\nPrediction Result:")
        print(json.dumps(result, indent=2))
    else:
        # Run test demonstration cases
        print("\n--- Running Built-In Demonstration Inference Cases ---")
        demo_cases = [
            {
                "Profile": "Case 1: Prime Low-Risk Applicant (Graduate, High Income, Good Credit)",
                "Data": {
                    "Gender": "Male",
                    "Married": "Yes",
                    "Dependents": "1",
                    "Education": "Graduate",
                    "Self_Employed": "No",
                    "ApplicantIncome": 8500,
                    "CoapplicantIncome": 3200,
                    "LoanAmount": 150.0,
                    "Loan_Amount_Term": 360.0,
                    "Credit_History": 1.0,
                    "Property_Area": "Semiurban",
                },
            },
            {
                "Profile": "Case 2: Subprime High-Risk Applicant (Poor Credit, High Debt Burden)",
                "Data": {
                    "Gender": "Female",
                    "Married": "No",
                    "Dependents": "2",
                    "Education": "Not Graduate",
                    "Self_Employed": "Yes",
                    "ApplicantIncome": 2200,
                    "CoapplicantIncome": 0,
                    "LoanAmount": 260.0,
                    "Loan_Amount_Term": 360.0,
                    "Credit_History": 0.0,
                    "Property_Area": "Rural",
                },
            },
            {
                "Profile": "Case 3: Marginal Applicant (Good Credit, Moderate Income, Urban)",
                "Data": {
                    "Gender": "Male",
                    "Married": "No",
                    "Dependents": "0",
                    "Education": "Graduate",
                    "Self_Employed": "No",
                    "ApplicantIncome": 4100,
                    "CoapplicantIncome": 1500,
                    "LoanAmount": 130.0,
                    "Loan_Amount_Term": 360.0,
                    "Credit_History": 1.0,
                    "Property_Area": "Urban",
                },
            },
        ]

        for case in demo_cases:
            print(f"\nEvaluating -> {case['Profile']}")
            res = predictor.predict(case["Data"])
            print(json.dumps(res[0], indent=2))


if __name__ == "__main__":
    main()

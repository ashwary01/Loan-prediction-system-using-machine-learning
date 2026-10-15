"""
Script to generate realistic loan_data.csv matching the classic 614-row Kaggle/Analytics Vidhya dataset.
Ensures realistic distributions, missing value rates, and credit risk correlations.
"""

import numpy as np
import pandas as pd

def generate_loan_dataset(n_samples: int = 614, random_state: int = 42) -> pd.DataFrame:
    np.random.seed(random_state)
    
    # 1. Loan_ID
    loan_ids = [f"LP00{1002 + i:04d}" for i in range(n_samples)]
    
    # 2. Gender (~81% Male, 19% Female, ~2.1% missing)
    gender_raw = np.random.choice(['Male', 'Female'], size=n_samples, p=[0.81, 0.19])
    gender = pd.Series(gender_raw, dtype=object)
    gender[np.random.choice(n_samples, size=13, replace=False)] = np.nan
    
    # 3. Married (~65% Yes, 35% No, ~0.5% missing)
    married_raw = np.random.choice(['Yes', 'No'], size=n_samples, p=[0.65, 0.35])
    married = pd.Series(married_raw, dtype=object)
    married[np.random.choice(n_samples, size=3, replace=False)] = np.nan
    
    # 4. Dependents ('0': 57%, '1': 17%, '2': 17%, '3+': 9%, ~2.5% missing)
    dep_raw = np.random.choice(['0', '1', '2', '3+'], size=n_samples, p=[0.58, 0.17, 0.17, 0.08])
    dependents = pd.Series(dep_raw, dtype=object)
    dependents[np.random.choice(n_samples, size=15, replace=False)] = np.nan
    
    # 5. Education ('Graduate': 78%, 'Not Graduate': 22%)
    education = np.random.choice(['Graduate', 'Not Graduate'], size=n_samples, p=[0.78, 0.22])
    
    # 6. Self_Employed ('No': 86%, 'Yes': 14%, ~5.2% missing)
    se_raw = np.random.choice(['No', 'Yes'], size=n_samples, p=[0.86, 0.14])
    self_employed = pd.Series(se_raw, dtype=object)
    self_employed[np.random.choice(n_samples, size=32, replace=False)] = np.nan
    
    # 7. ApplicantIncome (Log-normal distribution, median ~3800, right skewed)
    app_inc = np.random.lognormal(mean=8.2, sigma=0.65, size=n_samples)
    app_inc = np.round(np.clip(app_inc, 150, 81000)).astype(int)
    
    # 8. CoapplicantIncome (~45% zero income, log-normal for the rest)
    has_coapplicant = np.random.binomial(1, 0.55, size=n_samples)
    coapp_raw = np.random.lognormal(mean=7.5, sigma=0.8, size=n_samples)
    coapp_inc = np.where(has_coapplicant == 1, np.round(coapp_raw), 0.0)
    coapp_inc = np.clip(coapp_inc, 0, 41667)
    
    # 9. LoanAmount in thousands (~3.6% missing, median ~128)
    total_income = app_inc + coapp_inc
    base_loan = (total_income * 0.024) + np.random.normal(loc=25, scale=32, size=n_samples)
    loan_amount = np.round(np.clip(base_loan, 9.0, 700.0), 1)
    mask_la_nan = np.random.choice(n_samples, size=22, replace=False)
    loan_amount[mask_la_nan] = np.nan
    
    # 10. Loan_Amount_Term (Dominant: 360, others: 180, 120, 240, 300, 480, 84, 60, ~2.3% missing)
    terms = [360.0, 180.0, 480.0, 300.0, 240.0, 120.0, 84.0, 60.0]
    term_probs = [0.85, 0.07, 0.025, 0.02, 0.015, 0.01, 0.005, 0.005]
    loan_term = np.random.choice(terms, size=n_samples, p=term_probs)
    mask_term_nan = np.random.choice(n_samples, size=14, replace=False)
    loan_term[mask_term_nan] = np.nan
    
    # 11. Credit_History (1.0 vs 0.0, ~8.1% missing)
    credit_hist = np.random.choice([1.0, 0.0], size=n_samples, p=[0.84, 0.16])
    mask_ch_nan = np.random.choice(n_samples, size=50, replace=False)
    credit_hist[mask_ch_nan] = np.nan
    
    # 12. Property_Area ('Semiurban': 38%, 'Urban': 33%, 'Rural': 29%)
    property_area = np.random.choice(['Semiurban', 'Urban', 'Rural'], size=n_samples, p=[0.38, 0.33, 0.29])
    
    # 13. Loan_Status ('Y' or 'N')
    loan_status = []
    for i in range(n_samples):
        ch = credit_hist[i]
        inc = total_income[i]
        la = 130.0 if np.isnan(loan_amount[i]) else loan_amount[i]
        pa = property_area[i]
        edu = education[i]
        
        if np.isnan(ch):
            prob = 0.68
        elif ch == 1.0:
            prob = 0.79
            if pa == 'Semiurban':
                prob += 0.07
            if edu == 'Graduate':
                prob += 0.04
            # Debt burden
            if (la * 1000) / (inc + 1e-5) > 42:
                prob -= 0.18
        else:
            prob = 0.09
            if pa == 'Semiurban':
                prob += 0.05
            if inc > 12000 and la < 140:
                prob += 0.12
        
        prob = np.clip(prob, 0.03, 0.96)
        status = 'Y' if np.random.rand() < prob else 'N'
        loan_status.append(status)
        
    df = pd.DataFrame({
        'Loan_ID': loan_ids,
        'Gender': gender,
        'Married': married,
        'Dependents': dependents,
        'Education': education,
        'Self_Employed': self_employed,
        'ApplicantIncome': app_inc,
        'CoapplicantIncome': coapp_inc,
        'LoanAmount': loan_amount,
        'Loan_Amount_Term': loan_term,
        'Credit_History': credit_hist,
        'Property_Area': property_area,
        'Loan_Status': loan_status
    })
    
    return df

if __name__ == "__main__":
    df = generate_loan_dataset(n_samples=614, random_state=42)
    df.to_csv("loan_data.csv", index=False)
    print(f"Generated loan_data.csv with {len(df)} rows.")
    print("Class distribution:")
    print(df['Loan_Status'].value_counts(normalize=True))
    print("\nMissing values:")
    print(df.isnull().sum())

"""
create_architecture_diagram.py - Generates high-resolution system architecture diagram
for Loan Default Prediction System.
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch

def draw_system_architecture(output_path="assets/system_architecture.png"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fig, ax = plt.subplots(figsize=(16, 9), dpi=300)
    fig.patch.set_facecolor('#070a13')
    ax.set_facecolor('#070a13')
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis('off')

    # Color Palette
    c_blue = '#0ea5e9'
    c_indigo = '#6366f1'
    c_emerald = '#10b981'
    c_purple = '#8b5cf6'
    c_amber = '#f59e0b'
    c_dark_card = '#111827'
    c_border = '#374151'
    c_text_main = '#f8fafc'
    c_text_sub = '#94a3b8'

    # Title & Subtitle
    ax.text(8, 8.5, "LoanGuard AI — System Architecture & ML Pipeline",
            fontsize=20, fontweight='bold', color=c_text_main, ha='center', va='center')
    ax.text(8, 8.1, "End-to-End Supervised Learning Workflow: Ingestion, Feature Engineering, CV, & Serving",
            fontsize=11, color=c_text_sub, ha='center', va='center')

    # Helper function to draw rounded cards
    def add_card(x, y, w, h, title, items, badge_text, badge_color, border_color):
        # Card Background
        card = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08,rounding_size=0.18",
                              facecolor=c_dark_card, edgecolor=border_color, linewidth=1.6)
        ax.add_patch(card)

        # Header Badge
        badge = FancyBboxPatch((x + 0.2, y + h - 0.42), len(badge_text) * 0.13 + 0.3, 0.32,
                               boxstyle="round,pad=0.04,rounding_size=0.08",
                               facecolor=badge_color, edgecolor='none')
        ax.add_patch(badge)
        ax.text(x + 0.35, y + h - 0.26, badge_text,
                fontsize=7.5, fontweight='bold', color='#ffffff', ha='left', va='center')

        # Card Title
        ax.text(x + 0.2, y + h - 0.72, title,
                fontsize=11.5, fontweight='bold', color=c_text_main, ha='left', va='center')

        # Card Body Items
        for idx, item in enumerate(items):
            item_y = y + h - 1.15 - (idx * 0.42)
            ax.text(x + 0.25, item_y, "•", fontsize=11, color=border_color, ha='left', va='center')
            ax.text(x + 0.5, item_y, item, fontsize=8.8, color=c_text_sub, ha='left', va='center')

    # 1. Ingestion Layer
    ingestion_items = [
        "Source: loan_data.csv (614 rows)",
        "Demographics: Gender, Married, Dependents",
        "Financials: Applicant & Coapplicant Income",
        "Collateral: LoanAmount & Term",
        "Bureau Record: Credit_History (0.0 / 1.0)",
        "Property_Area & Target Loan_Status (Y/N)"
    ]
    add_card(0.6, 3.8, 3.2, 3.6, "1. Data Ingestion Layer", ingestion_items, "INGESTION", c_blue, '#0284c7')

    # 2. Feature Engineering Layer
    fe_items = [
        "FeatureEngineer(BaseEstimator)",
        "Total_Income = Applicant + Coapplicant",
        "Total_Income_Log & LoanAmount_Log",
        "EMI = (LoanAmount * 1000) / Term",
        "Debt-to-Income (DTI) = EMI / Income",
        "Balance_Income = Discretionary Surplus"
    ]
    add_card(4.3, 3.8, 3.4, 3.6, "2. Feature Engineering", fe_items, "DOMAIN LOGIC", c_indigo, '#4f46e5')

    # 3. Preprocessing ColumnTransformer
    prep_items = [
        "Scikit-Learn ColumnTransformer",
        "Numeric Sub-Pipeline:",
        "  - Median Imputation (Outlier Resilient)",
        "  - StandardScaler (Zero Mean, Unit Var)",
        "Categorical Sub-Pipeline:",
        "  - Mode Imputer + OneHotEncoder(drop='first')"
    ]
    add_card(8.2, 3.8, 3.4, 3.6, "3. Preprocessing Pipeline", prep_items, "TRANSFORMER", c_purple, '#7c3aed')

    # 4. Model Training & Tuning
    train_items = [
        "Stratified 80/20 Train-Test Split",
        "5-Fold Stratified Cross-Validation",
        "Logistic Regression (Linear Baseline)",
        "Decision Tree (Non-Linear Partitioning)",
        "Random Forest + GridSearchCV Tuning",
        "Champion: Tuned RF (200 Trees, Depth 8)"
    ]
    add_card(12.1, 3.8, 3.3, 3.6, "4. Training & CV Tuning", train_items, "ESTIMATION", c_amber, '#d97706')

    # Connecting Flow Arrows across top row
    arrow_props = dict(facecolor=c_blue, edgecolor='none', width=0.08, headwidth=0.35, headlength=0.3)
    ax.annotate('', xy=(4.2, 5.6), xytext=(3.9, 5.6), arrowprops=arrow_props)
    ax.annotate('', xy=(8.1, 5.6), xytext=(7.8, 5.6), arrowprops=arrow_props)
    ax.annotate('', xy=(12.0, 5.6), xytext=(11.7, 5.6), arrowprops=arrow_props)

    # 5. Bottom Row: Evaluation & Serving
    eval_items = [
        "Holdout Test Accuracy: 82.93% | ROC-AUC: 0.775",
        "Defaulter Precision: 86.7% (Low False Approvals)",
        "Defaulter Recall Prioritization (Cost Asymmetry)",
        "Confusion Matrix: 89 Funded, 13 Defaulters Caught"
    ]
    add_card(2.4, 0.6, 5.2, 2.6, "5. Model Evaluation & Cost Analysis", eval_items, "VALIDATION", c_emerald, '#059669')

    serving_items = [
        "Export Artifact: models/loan_model.joblib (230 KB)",
        "REST API Server: GET /api/metrics, /api/presets",
        "Real-Time Inference: POST /api/predict",
        "Interactive Dashboard: SVG Risk Gauge & Presets"
    ]
    add_card(8.4, 0.6, 5.2, 2.6, "6. Serialization & Web Deployment", serving_items, "PRODUCTION", c_blue, '#0284c7')

    # Downward and Horizontal Connecting Arrows
    arrow_props_down = dict(facecolor='#10b981', edgecolor='none', width=0.08, headwidth=0.35, headlength=0.3)
    ax.annotate('', xy=(13.75, 3.3), xytext=(13.75, 3.7), arrowprops=arrow_props_down)
    ax.annotate('', xy=(11.1, 1.9), xytext=(13.75, 3.3),
                arrowprops=dict(arrowstyle="->", color='#10b981', lw=2.2, connectionstyle="angle,angleA=0,angleB=90,rad=10"))

    ax.annotate('', xy=(7.7, 1.9), xytext=(8.3, 1.9),
                arrowprops=dict(arrowstyle="<-", color='#0ea5e9', lw=2.2))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
    plt.close()
    print(f"Architecture diagram successfully generated at: {output_path}")

if __name__ == "__main__":
    draw_system_architecture("assets/system_architecture.png")

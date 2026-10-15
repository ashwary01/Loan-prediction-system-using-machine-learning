"""
server.py - Lightweight REST API & Web Dashboard Server for Loan Default Prediction.
Serves the interactive frontend and exposes /api/predict, /api/metrics, and /api/presets.
"""

import http.server
import socketserver
import json
import os
import sys
import mimetypes

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.predict import LoanPredictor

PORT = 8000
WEB_DIR = os.path.join(os.path.dirname(__file__), "web")

# Initialize Inference Predictor
predictor = LoanPredictor(model_path="models/loan_model.joblib")

PRESET_PROFILES = {
    "prime": {
        "title": "Prime Tier (Low Risk)",
        "description": "Graduate with $8,500 income, coapplicant, moderate $140k loan, and clean credit history.",
        "data": {
            "Gender": "Male",
            "Married": "Yes",
            "Dependents": "1",
            "Education": "Graduate",
            "Self_Employed": "No",
            "ApplicantIncome": 8500,
            "CoapplicantIncome": 3200,
            "LoanAmount": 140.0,
            "Loan_Amount_Term": 360.0,
            "Credit_History": 1.0,
            "Property_Area": "Semiurban"
        }
    },
    "subprime": {
        "title": "Subprime (High Default Risk)",
        "description": "Poor credit record, low income ($2,200), high requested loan ($260k), self-employed in rural area.",
        "data": {
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
            "Property_Area": "Rural"
        }
    },
    "marginal": {
        "title": "Marginal Urban (Moderate)",
        "description": "Graduate professional with good credit, single income, applying for an urban property.",
        "data": {
            "Gender": "Male",
            "Married": "No",
            "Dependents": "0",
            "Education": "Graduate",
            "Self_Employed": "No",
            "ApplicantIncome": 4800,
            "CoapplicantIncome": 1200,
            "LoanAmount": 150.0,
            "Loan_Amount_Term": 360.0,
            "Credit_History": 1.0,
            "Property_Area": "Urban"
        }
    },
    "small_business": {
        "title": "Small Business Owner",
        "description": "Self-employed applicant with $6,200 income, no co-signer, seeking $180k loan in semi-urban zone.",
        "data": {
            "Gender": "Male",
            "Married": "Yes",
            "Dependents": "2",
            "Education": "Graduate",
            "Self_Employed": "Yes",
            "ApplicantIncome": 6200,
            "CoapplicantIncome": 0,
            "LoanAmount": 180.0,
            "Loan_Amount_Term": 360.0,
            "Credit_History": 1.0,
            "Property_Area": "Semiurban"
        }
    }
}


class LoanAppHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=WEB_DIR, **kwargs)

    def do_GET(self):
        if self.path == "/api/presets":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps(PRESET_PROFILES).encode("utf-8"))
            return

        elif self.path == "/api/metrics":
            metrics_path = "models/loan_model_metrics.json"
            if os.path.exists(metrics_path):
                with open(metrics_path, "r") as f:
                    data = json.load(f)
            else:
                data = {"error": "Metrics not found"}
            
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps(data).encode("utf-8"))
            return

        # Serve static files from web/
        return super().do_GET()

    def do_POST(self):
        if self.path == "/api/predict":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length).decode("utf-8")
            
            try:
                applicant_data = json.loads(body)
                results = predictor.predict(applicant_data)
                prediction = results[0]

                # Compute supplementary financial ratios
                app_inc = float(applicant_data.get("ApplicantIncome", 0) or 0)
                coapp_inc = float(applicant_data.get("CoapplicantIncome", 0) or 0)
                tot_inc = app_inc + coapp_inc
                loan_amt = float(applicant_data.get("LoanAmount", 0) or 0)
                term = float(applicant_data.get("Loan_Amount_Term", 360) or 360)
                
                safe_term = term if term > 0 else 360.0
                emi = (loan_amt * 1000.0) / safe_term
                dti = (emi / (tot_inc + 1e-5)) * 100.0 if tot_inc > 0 else 100.0
                balance_income = tot_inc - emi

                prediction["Financial_Summary"] = {
                    "Total_Income": round(tot_inc, 2),
                    "Monthly_EMI": round(emi, 2),
                    "Debt_To_Income_Pct": round(dti, 1),
                    "Discretionary_Balance": round(balance_income, 2)
                }

                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(json.dumps(prediction).encode("utf-8"))
            except Exception as e:
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))
            return

        self.send_response(404)
        self.end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()


def start_server():
    os.makedirs(WEB_DIR, exist_ok=True)
    with socketserver.TCPServer(("", PORT), LoanAppHandler) as httpd:
        print(f"LoanGuard AI Server running live at: http://localhost:{PORT}")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down server.")
            httpd.server_close()


if __name__ == "__main__":
    start_server()

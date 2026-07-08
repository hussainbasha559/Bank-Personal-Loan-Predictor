import os
import joblib
import shap
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model = joblib.load(os.path.join(BASE_DIR, "models/loan_model.pkl"))
explainer = shap.Explainer(model)   # ✅ init once, not per prediction

FEATURE_ORDER = [
    "Age", "Experience", "Income",
    "Family", "Education",
    "Mortgage", "CD Account"
]

def predict(data):
    data_array = np.array([[data[f] for f in FEATURE_ORDER]]).astype(float)
    prediction = model.predict(data_array)[0]
    shap_values = explainer(data_array)
    return prediction, shap_values.values[0, :, 1], data_array                                     

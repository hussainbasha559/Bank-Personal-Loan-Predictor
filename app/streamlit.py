# Multi-page Streamlit App
import sys
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

# Force fresh imports - must be BEFORE any src imports
sys.dont_write_bytecode = True
for mod in ['src.predictor', 'src.agent', 'predictor', 'agent']:
    if mod in sys.modules:
        del sys.modules[mod]

import streamlit as st
import pandas as pd
import sqlite3
import shap
import matplotlib.pyplot as plt
import pickle

from src.predictor import predict, explainer  # ← explainer imported here
from src.agent import loan_agent

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_PATH = os.path.join(CURRENT_DIR, "..", "src")




st.set_page_config(page_title="Bank Loan App", page_icon="🏦", layout="centered")

# ── Load data ──────────────────────────────────────────────────
BASE = os.path.dirname(os.path.dirname(__file__))
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
feature_names = pickle.load(open(os.path.join(BASE_DIR, "../models/feature_names.pkl"), "rb"))

@st.cache_data
def load_data():
    # conn = sqlite3.connect(os.path.join(BASE, "data/bank_loan.db"))
    DB_PATH = os.path.join(os.path.dirname(__file__), "../data/bank_loan.db")
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM clean_data", conn)
    conn.close()
    return df

df = load_data()

# ── Sidebar navigation ─────────────────────────────────────────
st.sidebar.title("🏦 Bank Loan App")
page = st.sidebar.radio("Navigate", ["🏠 Home", "📊 Dashboard", "🔍 Prediction", "📋 Data View"])

# ══════════════════════════════════════════════════════════════
# PAGE 1: HOME
# ══════════════════════════════════════════════════════════════
if page == "🏠 Home":

    st.title("🏦 Bank Personal Loan Predictor")
    st.write("Welcome! This app helps predict whether a customer will accept a personal loan offer.")
    st.divider()

    

    st.subheader("📌 What this app does")
    st.write("- **Dashboard** → See key stats about the data")
    st.write("- **Prediction** → Enter customer details and get a loan decision")
    st.write("- **Data View** → Browse the clean dataset")
    st.divider()

    st.markdown("""
       💡 **Problem:** Banks contact thousands of customers for loan offers — but most say No.
                
       🤖 **Solution:** This ML model predicts WHO will say Yes — saving time and money.
        """)
    st.divider()

    st.subheader("📊 Model Info")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Customers", "5,000")
    col2.metric("Loan Acceptance", "9.6%")
    col3.metric("Model Accuracy", "98.4%")
    col4.metric("Top Factor", "Income")
    st.divider()

    st.subheader("⚙️ How It Works")
    col1, col2, col3 = st.columns(3)
    col1.info("**Step 1** 📋\nEnter customer details")
    col2.info("**Step 2** 🤖\nML model analyzes")
    col3.info("**Step 3** ✅\nGet loan decision")
    st.divider()

    st.subheader("🔑 Top Factors for Loan Approval")
    st.write("**Income**");        st.progress(0.588)
    st.write("**Education**");     st.progress(0.114)
    st.write("**Family Size**");   st.progress(0.072)
    st.write("**CD Account**");    st.progress(0.067)
    st.divider()

    st.warning("⚠️ Only 9.6% of customers accept loans — this model helps the bank target the RIGHT customers!")
# ══════════════════════════════════════════════════════════════
# PAGE 2: DASHBOARD
# ══════════════════════════════════════════════════════════════
elif page == "📊 Dashboard":
    st.title("📊 Dashboard")
    st.write("Key numbers from the dataset.")
    st.write("Quick overview of all 5,000 bank customers and loan patterns.")
    st.divider()

    # Key number cards
    total      = len(df)
    approved   = df["Personal Loan"].sum()
    rejected   = total - approved
    approval_rate = approved / total * 100
    avg_income_approved = df[df["Personal Loan"] == 1]["Income"].mean()
    avg_income_rejected = df[df["Personal Loan"] == 0]["Income"].mean()

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Customers", f"{total:,}")
    col2.metric("Loan Approved",   f"{approved:,}")
    col3.metric("Loan Rejected",   f"{rejected:,}")

    st.divider()

    col4, col5, col6 = st.columns(3)
    col4.metric("Approval Rate",         f"{approval_rate:.1f}%")
    col5.metric("Avg Income (Approved)", f"${avg_income_approved:.0f}K")
    col6.metric("Avg Income (Rejected)", f"${avg_income_rejected:.0f}K")

    st.divider()

    col7, col8, col9 = st.columns(3)
    cd_approval = df[df["CD Account"] == 1]["Personal Loan"].mean() * 100
    online_pct  = df["Online"].mean() * 100
    col7.metric("CD Account Holders → Loan", f"{cd_approval:.1f}%")
    col8.metric("Use Online Banking",         f"{online_pct:.1f}%")
    col9.metric("Avg Age",                    f"{df['Age'].mean():.0f} yrs")
    st.divider()

    st.subheader("📚 Loan Approval by Education")
    edu_data = df.groupby("Education")["Personal Loan"].mean() * 100
    edu_data.index = ["Undergrad", "Graduate", "Advanced"]
    st.bar_chart(edu_data)
    st.divider()

    st.subheader("💰 Income Distribution")
    st.bar_chart(df["Income"].value_counts().sort_index())
    st.divider()

    st.info("""
💡 **Key Insights**
    - High income customers (above $100K) are 3x more likely to accept
    - CD Account holders have 46% acceptance rate
    - Advanced degree holders approve the most
    """)

# ══════════════════════════════════════════════════════════════
# PAGE 3: PREDICTION
# ══════════════════════════════════════════════════════════════

elif page == "🔍 Prediction":
    st.title("🏦 Loan Approval AI System")
    st.write("Enter applicant details below:")
    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        age        = st.number_input("Age",                        min_value=18, max_value=80, value=35)
        experience = st.number_input("Work Experience (years)",    min_value=0,  max_value=50, value=10)
        income     = st.number_input("Annual Income ($K)",         min_value=1,  max_value=500, value=60)
        family     = st.selectbox("Family Size", [1, 2, 3, 4])
        # ccavg      = st.number_input("Monthly CC Spend ($K)",      min_value=0.0, max_value=10.0, value=1.0, step=0.1)

    with col2:
        education_map = {"UG": 1, "Graduate": 2, "Advanced": 3}
        education_label = st.selectbox("Education", list(education_map.keys()))
        education = education_map[education_label]
        mortgage   = st.number_input("Mortgage Amount ($K)",       min_value=0, max_value=700, value=0)
        # sec_acc    = st.selectbox("Securities Account?", [0, 1],   format_func=lambda x: "Yes" if x else "No")
        cd_acc     = st.selectbox("CD Account?",         [0, 1],   format_func=lambda x: "Yes" if x else "No")
    st.divider()
    
    if st.button("🔍 Predict Loan Eligibility", use_container_width=True):

        st.info("🤖 AI Agent is analyzing your profile...")

        data = {
            "Age": age,
            "Experience": experience,
            "Income": income,
            "Family": family,
            "Education": education,
            "Mortgage": mortgage,
            "CD Account": cd_acc
        }
        st.write("DEBUG DATA:", data)  
        result = loan_agent(data)

        st.divider()

        st.subheader("📊 Decision")
        if "Approved" in result["decision"]:
            st.success(result["decision"])
        else:
            st.error(result["decision"])

        st.subheader("📈 Confidence")
        st.write(f"{result['confidence']}%")

        st.subheader("🧠 Insights")
        for i in result["insights"]:
            st.write("•", i)

        st.subheader("💡 Suggestions")
        if result["suggestions"]:
            for s in result["suggestions"]:
                st.write("•", s)
        else:
            st.write("No suggestions needed 👍")

        shap_values = result["shap_values"]
        data_array = result["data_array"]

        st.write("### Explanation:")

        fig, ax = plt.subplots()
        shap.plots.waterfall(
            shap.Explanation(
                values=shap_values,           # now 1D, 7 values
                base_values=explainer.expected_value[1],  # class-1 base value
                data=data_array[0],
                feature_names=["Age","Experience","Income","Family","Education","Mortgage","CD Account"]
            ),
            show=False
        )
        st.pyplot(fig)
    # if st.button("🔍 Predict Loan Eligibility", use_container_width=True):
    #     payload = {
    #         "Age": age, "Experience": experience, "Income": income,
    #         "Family": family,  "Education": education,
    #         "Mortgage": mortgage, 
    #         "CD_Account": cd_acc, 
    #     }
    #     # API_URL = os.getenv("API_URL", "https://mini-project-yyii.onrender.com/predict")
    #     # res = requests.post(API_URL, json=payload)
    #     try:
    #         r   = res.json()
    #         st.write("API says:", r) 
    #         if r["prediction"] == 1:
    #             st.success(f"### {r['result']}")
    #         else:
    #             st.error(f"### {r['result']}")
    #         st.write(f"**Confidence:** {r['confidence']}")
    #     except Exception as e:
    #         st.error(f"API Error: {e}")

# ══════════════════════════════════════════════════════════════
# PAGE 4: DATA VIEW
# ══════════════════════════════════════════════════════════════
elif page == "📋 Data View":
    st.title("📋 Clean Dataset")
    st.write(f"Showing {len(df):,} rows × {len(df.columns)} columns")
    st.divider()

    # Filter by loan status
    filter_opt = st.selectbox("Filter by Loan Status", ["All", "Approved (1)", "Rejected (0)"])
    if filter_opt == "Approved (1)":
        st.dataframe(df[df["Personal Loan"] == 1].reset_index(drop=True), use_container_width=True)
    elif filter_opt == "Rejected (0)":
        st.dataframe(df[df["Personal Loan"] == 0].reset_index(drop=True), use_container_width=True)
    else:
        st.dataframe(df, use_container_width=True)
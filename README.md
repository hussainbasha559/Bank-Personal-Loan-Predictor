# 🏦 Explainable Loan Approval Prediction System

> An intelligent machine learning system that predicts loan approval and explains decisions using SHAP for full transparency.

---

## 📌 Problem Statement

Banks contact thousands of customers for personal loan offers, but only **~9.6% convert**, leading to wasted effort and cost.

### 🎯 Goal

Build a system that:

* Predicts **who is likely to accept a loan**
* Provides **clear explanations for each decision**
* Helps improve **targeting efficiency**

---

## 💡 Solution Overview

This project combines:

* 🤖 **Machine Learning Prediction**
* 🧠 **Explainable AI (SHAP)**
* 💬 **Human-readable insights & suggestions**
* 🌐 **Interactive Streamlit Web App**

👉 Instead of just predicting, the system explains *why* a decision was made.

---

## 🎯 What the Model Predicts

| Output          | Meaning                     |
| --------------- | --------------------------- |
| ✅ Loan Approved | Customer likely to accept   |
| ❌ Loan Rejected | Customer unlikely to accept |

---

## 🧠 Key Innovation (IMPORTANT FOR JUDGES)

Unlike traditional ML projects, this system includes:

✔ SHAP-based feature explanations
✔ Insight generation (increase/decrease impact)
✔ Smart suggestions for improvement
✔ Full transparency (no black-box decisions)

---

## 📊 Dataset Information

* **Rows:** 5,000 customers
* **Target:** Personal Loan (0/1)
* **Final Features Used:** 7

### Features:

* Age
* Experience
* Income
* Family
* Education
* Mortgage
* CD Account

---

## 🤖 Model Details

| Detail           | Value         |
| ---------------- | ------------- |
| Algorithm        | Random Forest |
| Accuracy         | **98.4%**     |
| Train/Test Split | 80/20         |
| Class Handling   | Balanced      |

---

## 🔍 Key Insights

* 💰 **Income (58.8%)** → strongest factor
* 🎓 Higher education → more acceptance
* 👨‍👩‍👧 Larger families → higher demand
* 💳 CD Account holders → significantly higher acceptance

---

## 🧠 Explainability (SHAP)

Each prediction includes:

* Feature contribution (positive/negative)
* Waterfall visualization
* Decision reasoning

👉 Example:

* Income ↑ increases approval
* Age ↓ decreases approval

---

## 🌐 Application (Streamlit)

### Pages:

| Page          | Description                   |
| ------------- | ----------------------------- |
| 🏠 Home       | Overview                      |
| 📊 Dashboard  | Data insights                 |
| 🔍 Prediction | Live prediction + explanation |
| 📋 Data View  | Dataset exploration           |

---

## ⚙️ How It Works

1. User inputs customer details
2. Model predicts approval
3. SHAP calculates feature impact
4. Agent converts results into:

   * Insights
   * Suggestions

---

## 🚀 How to Run

```bash
pip install -r requirements.txt
```

```bash
uvicorn app.api:app --reload
```

```bash
streamlit run app/streamlit.py
```

---



## 🏆 Why This Project Stands Out

✔ Not just prediction — **explainable AI system**
✔ Real-world business use case
✔ Interactive and user-friendly
✔ Combines ML + UI + interpretability

---


## 📌 Conclusion

This project demonstrates how machine learning can be made **transparent, interpretable, and actionable**, especially in high-stakes domains like banking.



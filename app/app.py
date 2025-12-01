import streamlit as st
import pandas as pd
import numpy as np
from joblib import load
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="Churn Prediction App", layout="wide")

st.title("🔮 Customer Churn Prediction & SHAP Analysis")

# ----------------------------------------------------
# LOAD MODEL & TRANSFORMERS
# ----------------------------------------------------
clf = load(r"C:\Users\LENOVO\Documents\GitHub\ML_PowerBI\models\clf_final.joblib")
scaler = load(r"C:\Users\LENOVO\Documents\GitHub\ML_PowerBI\models\scaler_final.joblib")
ohe = load(r"C:\Users\LENOVO\Documents\GitHub\ML_PowerBI\models\ohe_encoder.joblib")

# SHAP necesita los mismos features del entrenamiento
X_train_final = pd.read_csv(
    r"C:\Users\LENOVO\Documents\GitHub\ML_PowerBI\data\processed\X_train_final.csv"
)

# ----------------------------------------------------
# DEFINE COLUMNS
# ----------------------------------------------------
numerical_cols = [
    'AccountAge','MonthlyCharges','TotalCharges','ViewingHoursPerWeek',
    'AverageViewingDuration','ContentDownloadsPerMonth','SupportTicketsPerMonth','WatchlistSize'
]

categorical_cols = [
    "SubscriptionType", "PaymentMethod", "PaperlessBilling",
    "ContentType", "MultiDeviceAccess", "DeviceRegistered",
    "Gender", "ParentalControl", "GenrePreference"
]

# ----------------------------------------------------
# SIDEBAR USER INPUTS
# ----------------------------------------------------
st.sidebar.header("🔧 Input del cliente")

user_num = {}
for col in numerical_cols:
    user_num[col] = st.sidebar.number_input(col, min_value=0.0, value=10.0)

user_cat = {}
user_cat["SubscriptionType"] = st.sidebar.selectbox(
    "SubscriptionType", ["Basic","Standard","Premium"]
)
user_cat["PaymentMethod"] = st.sidebar.selectbox(
    "PaymentMethod", ["Credit card", "Electronic check", "Mailed check"]
)
user_cat["PaperlessBilling"] = st.sidebar.selectbox(
    "PaperlessBilling", ["Yes","No"]
)
user_cat["ContentType"] = st.sidebar.selectbox(
    "ContentType", ["Movies","TV Shows"]
)
user_cat["MultiDeviceAccess"] = st.sidebar.selectbox(
    "MultiDeviceAccess", ["Yes","No"]
)
user_cat["DeviceRegistered"] = st.sidebar.selectbox(
    "DeviceRegistered", ["Mobile","TV","Tablet"]
)
user_cat["Gender"] = st.sidebar.selectbox(
    "Gender", ["Male","Female"]
)
user_cat["ParentalControl"] = st.sidebar.selectbox(
    "ParentalControl", ["Yes","No"]
)
user_cat["GenrePreference"] = st.sidebar.selectbox(
    "GenrePreference", ["Action","Drama","Comedy","Family","Sports"]
)

input_df = pd.DataFrame({**user_num, **user_cat}, index=[0])

st.write("### 🧾 Datos ingresados")
st.dataframe(input_df)

# ----------------------------------------------------
# TRANSFORM INPUT
# ----------------------------------------------------

# 1. OneHotEncoder
input_ohe = ohe.transform(input_df[categorical_cols])
ohe_cols = ohe.get_feature_names_out(categorical_cols)
input_ohe_df = pd.DataFrame(input_ohe, columns=ohe_cols)

# 2. Scaling numéricas
input_scaled = scaler.transform(input_df[numerical_cols])
input_scaled_df = pd.DataFrame(input_scaled, columns=numerical_cols)

# 3. Concatenar
input_final = pd.concat([input_scaled_df, input_ohe_df], axis=1)

# Alineación con entrenamiento (muy importante)
missing_cols = set(X_train_final.columns) - set(input_final.columns)
for col in missing_cols:
    input_final[col] = 0  # crear columnas faltantes en 0

input_final = input_final[X_train_final.columns]

# ----------------------------------------------------
# PREDICTION
# ----------------------------------------------------
prob_churn = clf.predict_proba(input_final)[0,1]
prediction = int(prob_churn >= 0.3)

st.markdown("## 🔍 Predicción")
st.write(f"**Probabilidad de churn:** `{prob_churn:.3f}`")

if prediction == 1:
    st.error("⚠️ El modelo predice que **EXISTE riesgo de churn**.")
else:
    st.success("✔️ El cliente probablemente **se queda**.")

# ----------------------------------------------------
# SHAP PLOT
# ----------------------------------------------------
st.markdown("---")
st.subheader("📊 Explicación del modelo (SHAP)")

explainer = shap.LinearExplainer(
    clf,
    X_train_final,
    feature_perturbation="independent"
)

shap_values = explainer.shap_values(input_final)

fig, ax = plt.subplots(figsize=(10, 3))
shap.summary_plot(shap_values, input_final, plot_type="bar", show=False)
st.pyplot(fig)
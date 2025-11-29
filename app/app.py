import streamlit as st
import pandas as pd
import numpy as np
from joblib import load
import shap
import matplotlib.pyplot as plt

# Cargar modelo y scaler
clf = load("../models/clf_final.joblib")
scaler = load("../models/scaler_final.joblib")

# Título de la app
st.title("Predicción de Churn de Clientes")
st.write("Esta app predice la probabilidad de que un cliente se desuscriba del servicio y muestra la explicación con SHAP.")

# Input del cliente (manual)
st.sidebar.header("Ingrese datos del cliente")

def user_input_features():
    AccountAge = st.sidebar.number_input("Account Age (meses)", min_value=0, value=12)
    MonthlyCharges = st.sidebar.number_input("Monthly Charges", min_value=0.0, value=12.0)
    TotalCharges = st.sidebar.number_input("Total Charges", min_value=0.0, value=100.0)
    # Aquí agregar más variables que tu modelo requiera, numéricas y dummies
    # Por simplicidad se omiten dummies; podrías agregar selectboxes para variables categóricas
    data = {'AccountAge': AccountAge,
            'MonthlyCharges': MonthlyCharges,
            'TotalCharges': TotalCharges}
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Escalar variables numéricas
input_scaled = scaler.transform(input_df)

# Predecir probabilidad de churn
prob_churn = clf.predict_proba(input_scaled)[:,1][0]
pred_class = clf.predict(input_scaled)[0]

st.write(f"**Probabilidad de Churn:** {prob_churn:.2f}")
st.write(f"**Predicción:** {'Churn' if pred_class==1 else 'Activo'}")

# Explicación SHAP
explainer = shap.Explainer(clf, input_scaled)
shap_values = explainer(input_scaled)

st.subheader("Explicación SHAP de la predicción")
fig, ax = plt.subplots()
shap.plots.bar(shap_values, show=False)
st.pyplot(fig)
import streamlit as st
import pandas as pd
import numpy as np
import os
from joblib import load
import shap

# -------------------------
# 1. Configuración de rutas
# -------------------------
BASE_DIR = os.path.dirname(__file__)            # Carpeta donde está app.py
MODEL_PATH = os.path.join(BASE_DIR, "../models/clf_final.joblib")
SCALER_PATH = os.path.join(BASE_DIR, "../models/scaler_final.joblib")

# -------------------------
# 2.  Cargar modelo y scaler
# -------------------------
clf = load(MODEL_PATH)
scaler = load(SCALER_PATH)

# -------------------------
# 3. Sidebar para inputs de usuario
# -------------------------
st.sidebar.header("Parámetros del cliente")
# Ejemplo: si tus columnas OHE son SubscriptionType_Basic, SubscriptionType_Premium, etc.
user_input = {
    'MonthlyCharges': st.sidebar.number_input("Monthly Charges", min_value=0.0, value=15.0),
    'TotalCharges': st.sidebar.number_input("Total Charges", min_value=0.0, value=300.0),
    'AccountAge': st.sidebar.number_input("Account Age (months)", min_value=0, value=12),
    # Variables categóricas
    'SubscriptionType_Basic': st.sidebar.checkbox("Basic", value=False),
    'SubscriptionType_Premium': st.sidebar.checkbox("Premium", value=False),
    'SubscriptionType_Standard': st.sidebar.checkbox("Standard", value=False),
    # Agrega más variables según tu OHE
}

# -------------------------
# 4. Crear DataFrame con todas las columnas de entrenamiento
# -------------------------
# Debes tener las columnas finales del X_train_final de tu entrenamiento
columns_train = [...]  # Lista de todas las columnas numéricas + OHE usadas en entrenamiento

input_df = pd.DataFrame(columns=columns_train)
for col, val in user_input.items():
    if col in input_df.columns:
        input_df.loc[0, col] = val

# Rellenar NaN con 0 (para categorías no marcadas)
input_df = input_df.fillna(0)

# -------------------------
# 5. Escalar variables numéricas
# -------------------------
input_scaled = scaler.transform(input_df)  # Solo escala las columnas numéricas si tu scaler fue entrenado así

# -------------------------
# 6. Predicción
# -------------------------
prob_churn = clf.predict_proba(input_scaled)[:,1][0]  # Probabilidad de churn
st.subheader("Probabilidad de churn")
st.write(f"{prob_churn:.2%}")

# -------------------------
# 7. SHAP
# -------------------------
explainer = shap.LinearExplainer(clf, input_scaled)  # Si usaste LogisticRegression
shap_values = explainer.shap_values(input_scaled)

st.subheader("Impacto de las variables (SHAP)")
shap.initjs()
shap.force_plot(explainer.expected_value, shap_values, input_df, matplotlib=True, show=True)
st.pyplot(bbox_inches='tight')
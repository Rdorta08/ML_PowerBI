import streamlit as st
import pandas as pd
import numpy as np
from joblib import load
import shap
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Churn Prediction App", layout="wide")
st.title("🔮 Customer Churn Prediction & SHAP Analysis")

# -----------------------------------------------------------------------------
# LOAD MODEL & PREPROCESSORS
# -----------------------------------------------------------------------------
clf = load(r"C:\Users\LENOVO\Documents\GitHub\ML_PowerBI\models\clf_final.joblib")
scaler = load(r"C:\Users\LENOVO\Documents\GitHub\ML_PowerBI\models\scaler_final.joblib")
ohe = load(r"C:\Users\LENOVO\Documents\GitHub\ML_PowerBI\models\ohe_encoder.joblib")

X_train_final = pd.read_csv(
    r"C:\Users\LENOVO\Documents\GitHub\ML_PowerBI\data\processed\X_train_final.csv"
)

# -----------------------------------------------------------------------------
# COLUMN DEFINITIONS
# -----------------------------------------------------------------------------
numerical_cols = scaler.feature_names_in_.tolist()
categorical_cols = ohe.feature_names_in_.tolist()

# Obtener categorías reales del OHE entrenado
ohe_categories = {
    col: list(ohe.categories_[i])
    for i, col in enumerate(categorical_cols)
}

# -----------------------------------------------------------------------------
# SIDEBAR USER INPUT
# -----------------------------------------------------------------------------
st.sidebar.header("🔧 Input del cliente")

user_num = {}
for col in numerical_cols:
    user_num[col] = st.sidebar.number_input(col, min_value=0.0, value=10.0)

user_cat = {}
for col in categorical_cols:
    user_cat[col] = st.sidebar.selectbox(col, ohe_categories[col])

# Crear input dataframe
input_df = pd.DataFrame({**user_num, **user_cat}, index=[0])

st.write("### 🧾 Datos ingresados")
st.dataframe(input_df)

# -----------------------------------------------------------------------------
# DATA TRANSFORMATION
# -----------------------------------------------------------------------------
try:
    # 1. OneHot
    input_ohe = ohe.transform(input_df[categorical_cols])
    ohe_cols = ohe.get_feature_names_out(categorical_cols)
    input_ohe_df = pd.DataFrame(input_ohe, columns=ohe_cols)

    # 2. Scale numerical
    input_scaled = scaler.transform(input_df[numerical_cols])
    input_scaled_df = pd.DataFrame(input_scaled, columns=numerical_cols)

    # 3. Combine
    input_final = pd.concat([input_scaled_df, input_ohe_df], axis=1)

    # 4. Align with training columns
    for col in X_train_final.columns:
        if col not in input_final.columns:
            input_final[col] = 0

    input_final = input_final[X_train_final.columns]

except Exception as e:
    st.error(f"Error en la transformación: {e}")
    st.stop()

# -----------------------------------------------------------------------------
# PREDICTION
# -----------------------------------------------------------------------------
prob_churn = clf.predict_proba(input_final)[0, 1]
prediction = int(prob_churn >= 0.3)

st.markdown("## 🔍 Predicción")
st.write(f"**Probabilidad de churn:** `{prob_churn:.3f}`")

if prediction == 1:
    st.error("⚠️ Riesgo de churn.")
else:
    st.success("✔️ El cliente se queda.")

# -----------------------------------------------------------------------------
# SHAP ANALYSIS (VERSIÓN ESTABLE)
# -----------------------------------------------------------------------------
st.markdown("---")
st.subheader("📊 Explicación del modelo (SHAP)")

# Crear explainer automático (funciona con LogisticRegression + OHE + scaler)
explainer = shap.Explainer(clf, X_train_final)

# Calcular shap values
shap_values = explainer(input_final)

# Dibujar gráfico SHAP
fig, ax = plt.subplots(figsize=(10, 3))
shap.summary_plot(
    shap_values.values,
    input_final,
    plot_type="bar",
    show=False
)

st.pyplot(plt.gcf())
plt.clf()
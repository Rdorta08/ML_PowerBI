import streamlit as st
import pandas as pd
import numpy as np
from joblib import load
import shap
import matplotlib.pyplot as plt

# -------------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------------
st.set_page_config(page_title="Churn Prediction App", layout="wide")
st.title("🔮 Customer Churn Prediction & SHAP Analysis")

# -------------------------------------------------------------------------
# LOAD MODEL & PREPROCESSORS
# -------------------------------------------------------------------------
clf = load(r"C:\Users\LENOVO\Documents\GitHub\ML_PowerBI\models\clf_final.joblib")
scaler = load(r"C:\Users\LENOVO\Documents\GitHub\ML_PowerBI\models\scaler_final.joblib")
ohe = load(r"C:\Users\LENOVO\Documents\GitHub\ML_PowerBI\models\ohe_encoder.joblib")

X_train_final = pd.read_csv(
    r"C:\Users\LENOVO\Documents\GitHub\ML_PowerBI\data\processed\X_train_final.csv"
)

# -------------------------------------------------------------------------
# COLUMN DEFINITIONS
# -------------------------------------------------------------------------
numerical_cols = scaler.feature_names_in_.tolist()
categorical_cols = ohe.feature_names_in_.tolist()

# Obtener categorías reales del OHE entrenado
ohe_categories = {
    col: list(ohe.categories_[i])
    for i, col in enumerate(categorical_cols)
}

# -------------------------------------------------------------------------
# SIDEBAR USER INPUT
# -------------------------------------------------------------------------
st.sidebar.header("🔧 Input del cliente")

# Inputs numéricos
user_num = {}
for col in numerical_cols:
    user_num[col] = st.sidebar.number_input(col, min_value=0.0, value=10.0)

# Inputs categóricos
user_cat = {}
for col in categorical_cols:
    user_cat[col] = st.sidebar.selectbox(col, ohe_categories[col])

# Umbral ajustable
st.sidebar.markdown("### ⚙️ Ajuste de umbral de riesgo")
threshold = st.sidebar.slider(
    "Umbral de clasificación",
    min_value=0.0,
    max_value=1.0,
    value=0.3,
    step=0.01
)

# Crear input dataframe
input_df = pd.DataFrame({**user_num, **user_cat}, index=[0])

st.write("### 🧾 Datos ingresados")
st.dataframe(input_df)

# -------------------------------------------------------------------------
# DATA TRANSFORMATION
# -------------------------------------------------------------------------
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

# -------------------------------------------------------------------------
# PREDICTION
# -------------------------------------------------------------------------
prob_churn = clf.predict_proba(input_final)[0, 1]
prediction = int(prob_churn >= threshold)

st.markdown("## 🔍 Predicción")
st.write(f"**Probabilidad de churn:** `{prob_churn:.3f}`")
st.write(f"**Umbral seleccionado:** `{threshold:.2f}`")

if prediction == 1:
    st.error("⚠️ Riesgo de churn.")
else:
    st.success("✔️ El cliente se queda.")

# -------------------------------------------------------------------------
# SHAP ANALYSIS
# -------------------------------------------------------------------------
st.markdown("---")
st.subheader("📊 Explicación del modelo (SHAP)")

# Crear explainer
explainer = shap.Explainer(clf, X_train_final)
shap_values = explainer(input_final)

# SHAP ANALYSIS CON TABS
# -------------------------------------------------------------------------
st.markdown("---")
st.subheader("📊 Explicación del modelo (SHAP)")

# Crear explainer
explainer = shap.Explainer(clf, X_train_final)
shap_values = explainer(input_final)

# Crear pestañas
tab1, tab2, tab3 = st.tabs(["SHAP Resumen", "Beeswarm Plot", "Dependence Plot Interactivo"])

# --------------------- TAB 1: Bar plot ---------------------
with tab1:
    st.header("📊 SHAP: Importancia de features")
    fig, ax = plt.subplots(figsize=(10,3))
    shap.summary_plot(
        shap_values.values,
        input_final,
        plot_type="bar",
        show=False
    )
    st.pyplot(fig)
    plt.clf()

# --------------------- TAB 2: Beeswarm ---------------------
with tab2:
    st.header("🐝 SHAP: Beeswarm Plot")
    fig, ax = plt.subplots(figsize=(10,5))
    shap.summary_plot(
        shap_values.values,
        input_final,
        feature_names=X_train_final.columns,
        plot_type="dot",
        show=False
    )
    st.pyplot(fig)
    plt.clf()

# --------------------- TAB 3: Dependence Plot Interactivo ---------------------
with tab3:
    st.header("📈 SHAP: Dependence Plot Interactivo")

    # ------------------- Dependence Plot -------------------
    st.subheader("Dependence Plot (tendencia general)")

    # Selección de feature
    feature_to_plot = st.selectbox("Selecciona feature principal", X_train_final.columns.tolist())
    interaction_feature = st.selectbox(
        "Feature para color (interacción)",
        ["auto"] + X_train_final.columns.tolist()
    )

    # Crear explainer sobre X_train_final (dataset de referencia)
    explainer_ref = shap.Explainer(clf, X_train_final)
    shap_values_ref = explainer_ref(X_train_final)

    fig, ax = plt.subplots(figsize=(8,5))
    shap.dependence_plot(
        feature_to_plot,
        shap_values_ref.values,
        X_train_final,
        interaction_index=interaction_feature if interaction_feature != "auto" else "auto",
        ax=ax,
        show=False
    )
    st.pyplot(fig)
    plt.clf()

    
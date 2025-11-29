import streamlit as st
import pandas as pd
import numpy as np
from joblib import load
import shap

# -----------------------------
# 1️⃣ Cargar modelos y transformadores
# -----------------------------
clf = load(r"C:\Users\LENOVO\Documents\GitHub\ML_PowerBI\models\clf_final.joblib")
scaler = load(r"C:\Users\LENOVO\Documents\GitHub\ML_PowerBI\models\scaler_final.joblib")
ohe_columns = load(r"C:\Users\LENOVO\Documents\GitHub\ML_PowerBI\models\ohe_columns.joblib")  # lista de columnas OHE generadas en train
X_train_final = pd.read_csv(r"C:\Users\LENOVO\Documents\GitHub\ML_PowerBI\data\processed\X_train_final.csv", index_col=0)

# -----------------------------
# 2️⃣ Inputs del usuario en sidebar
# -----------------------------
st.title("Predicción de Churn")
st.sidebar.header("Ingrese los datos del cliente")

# Variables numéricas
AccountAge = st.sidebar.number_input("Account Age (meses)", min_value=0)
MonthlyCharges = st.sidebar.number_input("Monthly Charges")
TotalCharges = st.sidebar.number_input("Total Charges")
ViewingHoursPerWeek = st.sidebar.number_input("Viewing Hours Per Week")
AverageViewingDuration = st.sidebar.number_input("Average Viewing Duration")
ContentDownloadsPerMonth = st.sidebar.number_input("Content Downloads Per Month")
SupportTicketsPerMonth = st.sidebar.number_input("Support Tickets Per Month")
WatchlistSize = st.sidebar.number_input("Watchlist Size")

# Variables categóricas
SubscriptionType = st.sidebar.selectbox("Subscription Type", ["Basic", "Standard", "Premium"])
PaymentMethod = st.sidebar.selectbox("Payment Method", ["Credit card", "Electronic check", "Bank transfer", "Mailed check"])
PaperlessBilling = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])
ContentType = st.sidebar.selectbox("Content Type", ["Movies", "TV Shows", "Mixed"])
MultiDeviceAccess = st.sidebar.selectbox("MultiDevice Access", ["Yes", "No"])
DeviceRegistered = st.sidebar.selectbox("Device Registered", ["Mobile", "Computer", "TV", "Tablet"])
GenrePreference = st.sidebar.selectbox("Genre Preference", ["Comedy", "Drama", "Action", "Fantasy", "Sci-Fi"])
Gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
ParentalControl = st.sidebar.selectbox("Parental Control", ["Yes", "No"])

# -----------------------------
# 3️⃣ Crear DataFrame del input
# -----------------------------
input_dict = {
    "AccountAge": AccountAge,
    "MonthlyCharges": MonthlyCharges,
    "TotalCharges": TotalCharges,
    "ViewingHoursPerWeek": ViewingHoursPerWeek,
    "AverageViewingDuration": AverageViewingDuration,
    "ContentDownloadsPerMonth": ContentDownloadsPerMonth,
    "SupportTicketsPerMonth": SupportTicketsPerMonth,
    "WatchlistSize": WatchlistSize,
    "SubscriptionType": SubscriptionType,
    "PaymentMethod": PaymentMethod,
    "PaperlessBilling": PaperlessBilling,
    "ContentType": ContentType,
    "MultiDeviceAccess": MultiDeviceAccess,
    "DeviceRegistered": DeviceRegistered,
    "GenrePreference": GenrePreference,
    "Gender": Gender,
    "ParentalControl": ParentalControl
}
input_df = pd.DataFrame([input_dict])

# -----------------------------
# 4️⃣ Escalar variables numéricas
# -----------------------------
numerical_cols = ['AccountAge', 'MonthlyCharges', 'TotalCharges',
                  'ViewingHoursPerWeek', 'AverageViewingDuration',
                  'ContentDownloadsPerMonth', 'SupportTicketsPerMonth',
                  'WatchlistSize']

input_scaled = pd.DataFrame(
    scaler.transform(input_df[numerical_cols]),
    columns=numerical_cols
)

# -----------------------------
# 5️⃣ One-hot encoding de variables categóricas
# -----------------------------
categorical_cols = ["SubscriptionType", "PaymentMethod", "PaperlessBilling",
                    "ContentType", "MultiDeviceAccess", "DeviceRegistered",
                    "GenrePreference", "Gender", "ParentalControl"]

input_encoded = pd.get_dummies(input_df[categorical_cols])
for col in ohe_columns:
    if col not in input_encoded.columns:
        input_encoded[col] = 0
input_encoded = input_encoded[ohe_columns]

# -----------------------------
# 6️⃣ Concatenar numéricas y categóricas
# -----------------------------
input_final = pd.concat([input_scaled, input_encoded], axis=1)

# -----------------------------
# 7️⃣ Predicción
# -----------------------------
prob_churn = clf.predict_proba(input_final)[:,1][0]
st.subheader(f"Probabilidad de Churn: {prob_churn:.2f}")

# -----------------------------
# 8️⃣ SHAP
# -----------------------------
st.header("Importancia de features (SHAP)")
explainer = shap.LinearExplainer(clf, X_train_final, feature_dependence="independent")
shap_values = explainer.shap_values(input_final)

shap.initjs()
st_shap_plot = shap.force_plot(explainer.expected_value[1], shap_values[1], input_final)
st.pyplot(bbox_inches='tight')  # mostrar el gráfico
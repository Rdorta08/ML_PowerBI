# Customer Churn Project

Este proyecto analiza el abandono de clientes (churn) en un servicio de suscripción, aplicando técnicas de limpieza de datos, imputación, feature engineering, modelado y explicación con SHAP.

## Estructura del proyecto

- `data/raw/` → Dataset original
- `data/processed/` → Datos limpios, codificados y listos para modelado
- `data/shap/` → Valores SHAP para análisis de variables
- `notebooks/` → Notebooks paso a paso del análisis
- `reports/` → Figuras y reportes generados
- `README.md` → Documentación del proyecto

## Objetivo

- Detectar clientes propensos a abandonar el servicio.
- Maximizar recall de churners.
- Explicar el modelo con SHAP para decisiones de negocio.

## Variables clave

| Variable                | Descripción                                     |
|-------------------------|-----------------------------------------------|
| AccountAge              | Antigüedad de la cuenta en meses              |
| MonthlyCharges          | Costo mensual del servicio                     |
| TotalCharges            | Costo total acumulado                           |
| SubscriptionType        | Tipo de suscripción (Basic, Standard, Premium)|
| PaymentMethod           | Método de pago del cliente                     |
| ...                     | ...                                           |
| Churn                   | 1 = abandonó, 0 = activo                       |


## Dataset disponible en Kagle
https://www.kaggle.com/datasets/safrin03/predictive-analytics-for-customer-churn-dataset
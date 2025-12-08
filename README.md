# Customer Churn Project

Este proyecto analiza el abandono de clientes (churn) en un servicio de suscripción, aplicando limpieza de datos, imputación, modelado predictivo y explicación con SHAP. Además, se integraron los resultados en Power BI para generar insights de negocio.

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

## Flujo de trabajo

1. Preprocesamiento y limpieza de datos:
- Imputación de valores faltantes en variables categóricas y numéricas.
- Imputación de PaymentMethod posterior a la partición train/test con Random Forest, evitando data leakage.

2. Modelado predictivo:

- Regresión Logística con escalado y codificación de variables.
- Balanceo de clases con SMOTE.
- Ajuste de hiperparámetros con GridSearchCV.

3. Interpretación:

- SHAP values para identificar features más influyentes en el churn.
- Feature importance para generar insights accionables en negocio.

4. Visualización:

Dashboard en Power BI con vistas de Executive Overview, Segmentación de clientes y ML Insights.

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
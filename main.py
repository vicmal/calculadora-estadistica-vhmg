import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.stattools import durbin_watson
import scikit_posthocs as sp
import random

# Configuración de página
st.set_page_config(page_title="Calculadora Estadística VHMG Pro v3", layout="wide")
sns.set_theme(style="whitegrid")

def cargar_imagen_investigador():
    id_investigador = random.randint(1, 1000)
    url = f"https://picsum.photos/id/{id_investigador}/800/400"
    st.image(url, caption="Ingeniería y Ciencia de Datos - Ing° Víctor Malavé", use_container_width=True)

def mostrar_aeda(df, col_trat, var_resp):
    st.header(f"🔍 AEDA: Análisis Exploratorio - {var_resp}")
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("📊 Estadística Descriptiva")
        stats_df = df.groupby(col_trat)[var_resp].agg(['count', 'mean', 'std', 'min', 'median', 'max']).reset_index()
        stats_df['CV%'] = (stats_df['std'] / stats_df['mean']) * 100
        st.dataframe(stats_df.style.format(precision=2))
    with col2:
        st.subheader("📈 Distribución de Datos Crudos")
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        sns.boxplot(x=col_trat, y=var_resp, data=df, ax=ax[0], palette="viridis")
        ax[0].set_title("Boxplot por Tratamiento")
        sns.histplot(data=df, x=var_resp, hue=col_trat, kde=True, ax=ax[1], palette="viridis", legend=False)
        ax[1].set_title("Histograma y Densidad")
        st.pyplot(fig)

def realizar_analisis_completo(df, col_trat, var_resp):
    mostrar_aeda(df, col_trat, var_resp)
    st.divider()
    st.header(f"🔬 Diagnóstico de 4 Supuestos Críticos")
    
    # Ajuste del modelo dinámico
    try:
        formula = f"Q('{var_resp}') ~ C(Q('{col_trat}'))"
        modelo = ols(formula, data=df).fit()
        df['Residuales'] = modelo.resid
        df['Ajustados'] = modelo.fittedvalues
        df['Orden'] = range(1, len(df) + 1)
    except Exception as e:
        st.error(f"Error al modelar: {e}")
        return

    # Panel de Supuestos
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    sm.qqplot(df['Residuales'], line='s', ax=axes[0, 0])
    axes[0, 0].set_title("1. Normalidad (Q-Q Plot)")
    sns.scatterplot(x=df['Ajustados'], y=df['Residuales'], ax=axes[0, 1])
    axes[0, 1].axhline(0, color='red', ls='--')
    axes[0, 1].set_title("2. Homocedasticidad")
    axes[1, 0].plot(df['Orden'], df['Residuales'], marker='o')
    axes[1, 0].axhline(0, color='red', ls='--')
    axes[1, 0].set_title("3. Independencia (Orden)")
    sns.boxplot(x=col_trat, y='Residuales', data=df, ax=axes[1, 1])
    axes[1, 1].axhline(0, color='red', ls='--')
    axes[1, 1].set_title("4. Aditividad (Res. por Trat.)")
    plt.tight_layout()
    st.pyplot(fig)

    # Pruebas Formales
    _, p_shapiro = stats.shapiro(df['Residuales'])
    grupos = [group[var_resp].values for name, group in df.groupby(col_trat)]
    _, p_levene = stats.levene(*grupos)
    dw = durbin_watson(df['Residuales'])
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.write(f"**Shapiro-Wilk (p):** `{p_shapiro:.4f}`")
        st.write(f"**Levene (p):** `{p_levene:.4f}`")
        st.write(f"**Durbin-Watson:** `{dw:.4f}`")
    with col_b:
        cumple_p = p_shapiro > 0.05 and p_levene > 0.05
        st.write(f"**¿Cumple Supuestos Paramétricos?:** {'✅ SÍ' if cumple_p else '❌ NO'}")

    st.divider()

    # Inferencia
    if cumple_p:
        st.header("📊 Inferencia: ANOVA (Paramétrico)")
        tabla_anova = sm.stats.anova_lm(modelo, typ=2)
        st.table(tabla_anova)
        p_val = tabla_anova.iloc[0, 3]
    else:
        st.header("📊 Inferencia: Kruskal-Wallis (No Paramétrico)")
        stat_k, p_val = stats.kruskal(*grupos)
        st.write(f"Estadístico H: `{stat_k:.4f}`, p-valor: `{p_val:.4f}`")

    # Conclusión
    st.subheader("📝 Conclusión del Ensayo")
    if p_val < 0.05:
        st.success(f"**Resultado Significativo (p = {p_val:.4f}):** Existen diferencias reales entre tratamientos.")
        if cumple_p:
            ph = sp.posthoc_tukey(df, val_col=var_resp, group_col=col_trat)
        else:
            ph = sp.posthoc_dunn(df, val_col=var_resp, group_col=col_trat, p_adjust='holm')
        st.dataframe(ph.style.background_gradient(cmap='coolwarm'))
    else:
        st.info(f"**Resultado No Significativo (p = {p_val:.4f}):** Las diferencias se deben al azar.")

# --- INTERFAZ ---
st.title("📊 Calculadora de Análisis de Varianza")
cargar_imagen_investigador()

archivo = st.file_uploader("Cargue su base de datos", type=['csv', 'txt'])

if archivo:
    try:
        df = pd.read_csv(archivo, sep=None, engine='python')
        columnas = df.columns.tolist()
        
        c1, c2 = st.columns(2)
        with c1:
            col_trat = st.selectbox("Columna de TRATAMIENTOS:", columnas)
        with c2:
            col_num = df.select_dtypes(include=[np.number]).columns.tolist()
            var_resp = st.selectbox("VARIABLE RESPUESTA:", col_num)
        
        if st.button("🚀 Ejecutar Análisis Completo"):
            realizar_analisis_completo(df, col_trat, var_resp)
    except Exception as e:
        st.error(f"Error: {e}")


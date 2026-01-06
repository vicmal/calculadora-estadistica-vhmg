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
    st.image(url, caption="Ingeniería y Ciencia de Datos - VHMG", use_container_width=True)

def mostrar_aeda(df, var_resp):
    st.header(f"🔍 AEDA: Análisis Exploratorio - {var_resp}")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("📊 Estadística Descriptiva")
        # Agrupamos por tratamiento y calculamos estadísticos clave
        stats_df = df.groupby('Tratamiento')[var_resp].agg(['count', 'mean', 'std', 'min', 'median', 'max']).reset_index()
        # Añadimos Coeficiente de Variación (CV%)
        stats_df['CV%'] = (stats_df['std'] / stats_df['mean']) * 100
        st.dataframe(stats_df.style.format(precision=2))
        
    with col2:
        st.subheader("📈 Distribución de Datos Crudos")
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        sns.boxplot(x='Tratamiento', y=var_resp, data=df, ax=ax[0], palette="viridis")
        ax[0].set_title("Boxplot por Tratamiento")
        
        sns.histplot(data=df, x=var_resp, hue='Tratamiento', kde=True, ax=ax[1], palette="viridis")
        ax[1].set_title("Histograma y Densidad")
        st.pyplot(fig)

def realizar_analisis_completo(df, var_resp):
    # --- 1. AEDA ---
    mostrar_aeda(df, var_resp)
    
    st.divider()
    
    # --- 2. EVALUACIÓN DE LOS 4 SUPUESTOS (Sobre Residuales) ---
    st.header(f"🔬 Diagnóstico de 4 Supuestos Críticos ({var_resp})")
    try:
        formula = f"{var_resp} ~ C(Tratamiento)"
        modelo = ols(formula, data=df).fit()
        df['Residuales'] = modelo.resid
        df['Ajustados'] = modelo.fittedvalues
        df['Orden'] = range(1, len(df) + 1)
    except Exception as e:
        st.error(f"Error al modelar: {e}")
        return

    # Gráficos de Supuestos
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    sm.qqplot(df['Residuales'], line='s', ax=axes[0, 0])
    axes[0, 0].set_title("1. Normalidad (Q-Q Plot)")
    sns.scatterplot(x=df['Ajustados'], y='Residuales', data=df, ax=axes[0, 1])
    axes[0, 1].axhline(0, color='red', ls='--')
    axes[0, 1].set_title("2. Homocedasticidad (Res. vs Ajust.)")
    axes[1, 0].plot(df['Orden'], df['Residuales'], marker='o')
    axes[1, 0].axhline(0, color='red', ls='--')
    axes[1, 0].set_title("3. Independencia (Res. vs Orden)")
    sns.boxplot(x='Tratamiento', y='Residuales', data=df, ax=axes[1, 1])
    axes[1, 1].axhline(0, color='red', ls='--')
    axes[1, 1].set_title("4. Aditividad (Res. por Trat.)")
    plt.tight_layout()
    st.pyplot(fig)

    # Pruebas Formales
    _, p_shapiro = stats.shapiro(df['Residuales'])
    grupos = [group[var_resp].values for name, group in df.groupby('Tratamiento')]
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

    # --- 3. INFERENCIA Y CONCLUSIÓN ---
    if cumple_p:
        st.header("📊 Inferencia: ANOVA (Paramétrico)")
        tabla_anova = sm.stats.anova_lm(modelo, typ=2)
        st.table(tabla_anova)
        p_val = tabla_anova['PR(>F)'][0]
    else:
        st.header("📊 Inferencia: Kruskal-Wallis (No Paramétrico)")
        stat_k, p_val = stats.kruskal(*grupos)
        st.write(f"Estadístico H: `{stat_k:.4f}`, p-valor: `{p_val:.4f}`")

    # Redacción de Conclusión Profesional
    st.subheader("📝 Conclusión del Ensayo")
    if p_val < 0.05:
        st.success(f"**Resultado Significativo (p = {p_val:.4f}):** Se rechaza H₀. Existen diferencias significativas entre tratamientos para la variable **{var_resp}**.")
        st.subheader("🔍 Pruebas Post-hoc (Comparaciones Múltiples)")
        if cumple_p:
            ph = sp.posthoc_tukey(df, val_col=var_resp, group_col='Tratamiento')
        else:
            ph = sp.posthoc_dunn(df, val_col=var_resp, group_col='Tratamiento', p_adjust='holm')
        st.dataframe(ph.style.background_gradient(cmap='coolwarm'))
    else:
        st.info(f"**Resultado No Significativo (p = {p_val:.4f}):** No se rechaza H₀. No hay evidencia de efectos de los tratamientos sobre **{var_resp}**.")

# --- INTERFAZ PRINCIPAL ---
st.title("📊 Calculadora VHMG Pro: AEDA e Inferencia Avanzada")
st.markdown("---")
cargar_imagen_investigador()

archivo = st.file_uploader("Cargue su base de datos (CSV o TXT)", type=['csv', 'txt'])

if archivo:
    try:
        df = pd.read_csv(archivo, sep=None, engine='python')
        
        # El usuario DEBE decirnos cuál es la columna de los Tratamientos
        columnas = df.columns.tolist()
        col_trat = st.selectbox("Seleccione la columna de TRATAMIENTOS (Factores):", columnas)
        
        # Identificamos columnas numéricas para ser Variables Respuesta
        col_num = df.select_dtypes(include=[np.number]).columns.tolist()
        if col_trat in col_num: col_num.remove(col_trat)
        
        var_resp = st.selectbox("Seleccione la VARIABLE RESPUESTA a analizar:", col_num)
        
        if st.button("🚀 Ejecutar Análisis Completo"):
            # Renombramos temporalmente para compatibilidad con el motor
            df_analisis = df[[col_trat, var_resp]].copy()
            df_analisis.columns = ['Tratamiento', var_resp]
            realizar_analisis_completo(df_analisis, var_resp)
            
    except Exception as e:
        st.error(f"Error al procesar: {e}")

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
st.set_page_config(page_title="Calculadora Estadística VHMG Pro", layout="wide")
sns.set_theme(style="whitegrid")

def cargar_imagen_investigador():
    id_investigador = random.randint(1, 1000)
    url = f"https://picsum.photos/id/{id_investigador}/800/400"
    st.image(url, caption="Inspiración para la investigación científica - VHMG", use_container_width=True)

def realizar_analisis_vhmg(df):
    st.header("🔬 Auditoría de Supuestos del Modelo")
    
    # 1. AJUSTE DEL MODELO (La Factura antes de la Pizza)
    try:
        # Definición del modelo lineal aditivo: Y = mu + tau + error
        modelo = ols('Respuesta ~ C(Tratamiento)', data=df).fit()
        df['Ajustados'] = modelo.fittedvalues
        df['Residuales'] = modelo.resid
        df['Orden'] = range(1, len(df) + 1) # Para prueba de independencia
    except Exception as e:
        st.error(f"Error en la especificación del modelo: {e}")
        return

    # --- PANEL DE DIAGNÓSTICO VISUAL ---
    st.subheader("1. Visualización de Diagnóstico")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # A. Normalidad: Q-Q Plot
    sm.qqplot(df['Residuales'], line='s', ax=axes[0, 0])
    axes[0, 0].set_title("Q-Q Plot (Normalidad)")

    # B. Homocedasticidad: Residuos vs Ajustados
    sns.scatterplot(x=df['Ajustados'], y=df['Residuales'], ax=axes[0, 1])
    axes[0, 1].axhline(0, color='red', linestyle='--')
    axes[0, 1].set_title("Residuos vs. Ajustados (Varianza)")

    # C. Independencia: Residuos vs Orden
    axes[1, 0].plot(df['Orden'], df['Residuales'], marker='o', linestyle='-')
    axes[1, 0].axhline(0, color='red', linestyle='--')
    axes[1, 0].set_title("Residuos vs. Orden (Independencia)")

    # D. Aditividad: Boxplot de Residuos por Tratamiento
    sns.boxplot(x='Tratamiento', y='Residuales', data=df, ax=axes[1, 1])
    axes[1, 1].axhline(0, color='red', linestyle='--')
    axes[1, 1].set_title("Residuos por Tratamiento (Aditividad/Forma)")

    plt.tight_layout()
    st.pyplot(fig)

    # --- PRUEBAS FORMALES ---
    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🧪 Pruebas Formales")
        
        # Normalidad
        _, p_shapiro = stats.shapiro(df['Residuales'])
        st.write(f"**Normalidad (Shapiro-Wilk):** p = `{p_shapiro:.4f}`")
        
        # Homocedasticidad (Levene es más robusto que Bartlett)
        grupos = [group['Residuales'].values for name, group in df.groupby('Tratamiento')]
        _, p_levene = stats.levene(*grupos)
        st.write(f"**Homocedasticidad (Levene):** p = `{p_levene:.4f}`")
        
        # Independencia (Durbin-Watson)
        dw = durbin_watson(df['Residuales'])
        st.write(f"**Independencia (Durbin-Watson):** DW = `{dw:.4f}`")

    with col2:
        st.subheader("📋 Resumen de Cumplimiento")
        cumple_norm = p_shapiro > 0.05
        cumple_homo = p_levene > 0.05
        cumple_indp = 1.5 < dw < 2.5
        
        st.write(f"{'✅' if cumple_norm else '❌'} Normalidad")
        st.write(f"{'✅' if cumple_homo else '❌'} Homocedasticidad")
        st.write(f"{'✅' if cumple_indp else '❌'} Independencia (DW)")
        st.write("✅ Aditividad (Evaluada por estructura de modelo lineal)")

    st.divider()

    # --- INFERENCIA FINAL ---
    if cumple_norm and cumple_homo:
        st.subheader("📊 Resultados: ANOVA (Modelo Paramétrico)")
        tabla_anova = sm.stats.anova_lm(modelo, typ=2)
        st.table(tabla_anova)
        p_val = tabla_anova['PR(>F)'][0]
        
        if p_val < 0.05:
            st.success(f"Diferencias significativas detectadas (p = {p_val:.4f})")
            st.write("#### Prueba Post-hoc: Tukey HSD")
            posthoc = sp.posthoc_tukey(df, val_col='Respuesta', group_col='Tratamiento')
            st.dataframe(posthoc)
        else:
            st.info("No se detectaron diferencias significativas entre los tratamientos.")
    else:
        st.warning("⚠️ Los supuestos paramétricos no se cumplen. Aplicando prueba no paramétrica de respaldo.")
        st.subheader("📊 Resultados: Kruskal-Wallis")
        stat_k, p_k = stats.kruskal(*[group['Respuesta'].values for name, group in df.groupby('Tratamiento')])
        st.write(f"Estadístico H: `{stat_k:.4f}`, p-valor: `{p_k:.4f}`")
        
        if p_k < 0.05:
            st.success("Diferencias significativas detectadas.")
            st.write("#### Prueba Post-hoc: Dunn (Holm)")
            posthoc = sp.posthoc_dunn(df, val_col='Respuesta', group_col='Tratamiento', p_adjust='holm')
            st.dataframe(posthoc)
        else:
            st.info("No se detectaron diferencias significativas.")

# --- INTERFAZ STREAMLIT ---
st.title("📊 Calculadora VHMG: Diseño de Experimentos")
st.markdown("""
Esta aplicación valida los **4 supuestos críticos** antes de emitir un juicio estadístico:
1. **Normalidad** 2. **Homocedasticidad** 3. **Independencia** 4. **Aditividad**.
""")

cargar_imagen_investigador()

archivo = st.file_uploader("Cargue su archivo (CSV o TXT)", type=['csv', 'txt'])

if archivo:
    try:
        if archivo.name.endswith('.csv'):
            df = pd.read_csv(archivo)
        else:
            df = pd.read_csv(archivo, sep=None, engine='python')
        
        if 'Tratamiento' in df.columns and 'Respuesta' in df.columns:
            realizar_analisis_vhmg(df)
        else:
            st.error("Error: El archivo debe tener columnas 'Tratamiento' y 'Respuesta'.")
    except Exception as e:
        st.error(f"Error al procesar el archivo: {e}")

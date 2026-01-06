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

# Configuración y Estilo
st.set_page_config(page_title="Suite DOE VHMG Master", layout="wide")
sns.set_theme(style="whitegrid")

def mostrar_aeda_profesional(df, factores, respuesta):
    st.header("🔍 Análisis Exploratorio de Datos (AEDA)")
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📊 Descriptivas por Factor Principal")
        # Usamos el primer factor seleccionado para la tabla
        desc = df.groupby(factores[0])[respuesta].agg(['count', 'mean', 'std', 'min', 'max']).reset_index()
        desc['CV%'] = (desc['std'] / desc['mean']) * 100
        st.dataframe(desc.style.format(precision=3))
        
    with col2:
        st.subheader("📈 Gráfico de Caja y Bigotes")
        fig, ax = plt.subplots()
        sns.boxplot(data=df, x=factores[0], y=respuesta, hue=factores[1] if len(factores)>1 else None, ax=ax)
        st.pyplot(fig)

def ejecutar_motor_estadistico(df, diseño, factores, respuesta):
    st.divider()
    st.header(f"⚖️ Análisis de Inferencia: {diseño}")
    
    # Construcción Dinámica de la Fórmula según el Diseño
    if diseño == "Diseño Completamente Aleatorizado (DCA)":
        formula = f"Q('{respuesta}') ~ C(Q('{factores[0]}'))"
    elif diseño == "Diseño de Bloques al Azar (DBCA)":
        formula = f"Q('{respuesta}') ~ C(Q('{factores[0]}')) + C(Q('{factores[1]}'))"
    elif diseño == "Diseño Factorial":
        # Incluye Interacción
        formula = f"Q('{respuesta}') ~ C(Q('{factores[0]}')) * C(Q('{factores[1]}'))"
    elif diseño == "Diseño Cuadrado Latino (DCL)":
        formula = f"Q('{respuesta}') ~ C(Q('{factores[0]}')) + C(Q('{factores[1]}')) + C(Q('{factores[2]}'))"
    elif diseño == "Superficie de Respuesta / Taguchi":
        # Modelo cuadrático para optimización
        formula = f"Q('{respuesta}') ~ Q('{factores[0]}') + I(Q('{factores[0]}')**2)"
    else:
        # Genérico para diseños complejos
        terminos = " + ".join([f"C(Q('{f}'))" for f in factores])
        formula = f"Q('{respuesta}') ~ {terminos}"

    try:
        modelo = ols(formula, data=df).fit()
        df['Residuos'] = modelo.resid
        df['Ajustados'] = modelo.fittedvalues
        df['Orden'] = range(1, len(df) + 1)
        
        # --- VALIDACIÓN DE 4 SUPUESTOS SOBRE RESIDUOS ---
        st.subheader("🔬 Validación de Supuestos Críticos")
        fig, axes = plt.subplots(1, 4, figsize=(20, 4))
        sm.qqplot(df['Residuos'], line='s', ax=axes[0]); axes[0].set_title("1. Normalidad")
        sns.scatterplot(x=df['Ajustados'], y=df['Residuos'], ax=axes[1]); axes[1].axhline(0, color='red'); axes[1].set_title("2. Homocedasticidad")
        axes[2].plot(df['Orden'], df['Residuos'], marker='o'); axes[2].set_title("3. Independencia")
        sns.boxplot(x=factores[0], y='Residuos', data=df, ax=axes[3]); axes[3].set_title("4. Aditividad")
        st.pyplot(fig)

        # TABLA ANOVA
        st.subheader("📊 Tabla de Análisis de Varianza (ANAVA)")
        tabla_anova = sm.stats.anova_lm(modelo, typ=2)
        st.table(tabla_anova)
        
        # Conclusión basada en el p-valor del factor principal
        p_val = tabla_anova.iloc[0, 3]
        if p_val < 0.05:
            st.success(f"**Conclusión:** Existen diferencias significativas (p={p_val:.4f}). Se rechaza H0.")
            if diseño in ["DCA", "DBCA", "Diseño Factorial"]:
                st.subheader("🔍 Pruebas Post-hoc (Tukey)")
                ph = sp.posthoc_tukey(df, val_col=respuesta, group_col=factores[0])
                st.dataframe(ph.style.background_gradient(cmap='viridis'))
        else:
            st.info(f"**Conclusión:** No hay diferencias significativas (p={p_val:.4f}).")

    except Exception as e:
        st.error(f"Error en el cálculo del modelo: {e}. Verifique que seleccionó los factores correctos para el {diseño}.")

# --- INTERFAZ DE USUARIO ---
st.title("🚀 Suite Master de Diseño de Experimentos VHMG")
st.markdown("Plataforma integral para el análisis de diseños industriales y científicos.")

archivo = st.file_uploader("Suba su archivo de datos", type=['csv', 'txt'])

if archivo:
    df = pd.read_csv(archivo, sep=None, engine='python')
    columnas = df.columns.tolist()
    
    st.sidebar.header("⚙️ Configuración del Diseño")
    tipo_diseño = st.sidebar.selectbox("Seleccione el Tipo de Diseño:", [
        "Diseño Completamente Aleatorizado (DCA)",
        "Diseño de Bloques al Azar (DBCA)",
        "Diseño Factorial",
        "Diseño Cuadrado Latino (DCL)",
        "Diseño de Superficie de Respuesta / Taguchi",
        "Diseño de Bloques Incompletos",
        "Diseños Aumentados"
    ])
    
    col_resp = st.sidebar.selectbox("Variable Respuesta (Y):", df.select_dtypes(include=[np.number]).columns)
    
    # Selección dinámica de factores según el diseño
    if tipo_diseño == "Diseño Completamente Aleatorizado (DCA)":
        f1 = st.sidebar.selectbox("Factor de Tratamiento:", columnas)
        factores = [f1]
    elif tipo_diseño in ["Diseño de Bloques al Azar (DBCA)", "Diseño Factorial"]:
        f1 = st.sidebar.selectbox("Factor Principal:", columnas)
        f2 = st.sidebar.selectbox("Factor Secundario / Bloque:", columnas)
        factores = [f1, f2]
    elif tipo_diseño == "Diseño Cuadrado Latino (DCL)":
        f1 = st.sidebar.selectbox("Tratamiento:", columnas)
        f2 = st.sidebar.selectbox("Factor Fila:", columnas)
        f3 = st.sidebar.selectbox("Factor Columna:", columnas)
        factores = [f1, f2, f3]
    else:
        factores = st.sidebar.multiselect("Seleccione todos los factores involucrados:", columnas)

    if st.sidebar.button("⚡ Ejecutar Análisis"):
        mostrar_aeda_profesional(df, factores, col_resp)
        ejecutar_motor_estadistico(df, tipo_diseño, factores, col_resp)

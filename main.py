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

# Configuración y Estilo
st.set_page_config(page_title="Suite DOE VHMG Master v6", layout="wide")
sns.set_theme(style="whitegrid")

def cargar_imagen_investigador():
    id_investigador = random.randint(1, 1000)
    url = f"https://picsum.photos/id/{id_investigador}/800/400"
    st.image(url, caption="Ref. Autoría: Ing. Víctor Hugo Malavé Girón - Ingeniería y Ciencia de Datos", use_container_width=True)

def seccion_aeda(df, factor, respuesta):
    st.header(f"📊 Análisis Exploratorio de Datos (AEDA) - {respuesta}")
    
    col1, col2 = st.columns([1, 1.5])
    
    with col1:
        st.subheader("🔢 Estadísticas Descriptivas")
        # Cálculo de descriptivos
        desc = df.groupby(factor)[respuesta].agg(['count', 'mean', 'std', 'min', 'median', 'max']).reset_index()
        desc['CV%'] = (desc['std'] / desc['mean']) * 100
        st.dataframe(desc.style.format(precision=3).background_gradient(subset=['mean'], cmap='Blues'))
        st.info("**Nota:** Un CV% bajo indica mayor precisión en el experimento.")

    with col2:
        st.subheader("📈 Comportamiento de los Datos Crudos")
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        # Boxplot
        sns.boxplot(data=df, x=factor, y=respuesta, ax=ax[0], palette="Set2")
        ax[0].set_title("Distribución (Boxplot)")
        # Histograma
        sns.histplot(data=df, x=respuesta, hue=factor, kde=True, ax=ax[1], palette="Set2", legend=False)
        ax[1].set_title("Histograma y Densidad")
        st.pyplot(fig)

def prueba_aditividad_tukey(df, respuesta, modelo, factor):
    """Prueba de No-Aditividad de Tukey (1 Grado de Libertad)"""
    y_hat = modelo.fittedvalues
    df_aux = df.copy()
    df_aux['y_hat_sq'] = y_hat**2
    try:
        formula_aux = f"Q('{respuesta}') ~ C(Q('{factor}')) + y_hat_sq"
        modelo_aux = ols(formula_aux, data=df_aux).fit()
        return modelo_aux.pvalues['y_hat_sq']
    except:
        return 0.5

def realizar_auditoria_supuestos(df, respuesta, modelo, factores):
    st.header("🔬 Auditoría de los 4 Supuestos Críticos (Sobre Residuales)")
    residuos = modelo.resid
    ajustados = modelo.fittedvalues
    
    _, p_shapiro = stats.shapiro(residuos)
    grupos = [group[respuesta].values for name, group in df.groupby(factores[0])]
    _, p_levene = stats.levene(*grupos)
    dw_stat = durbin_watson(residuos)
    p_aditividad = prueba_aditividad_tukey(df, respuesta, modelo, factores[0])

    # Gráficos de Supuestos
    fig, axes = plt.subplots(1, 4, figsize=(20, 4))
    sm.qqplot(residuos, line='s', ax=axes[0]); axes[0].set_title("1. Normalidad (Q-Q)")
    sns.scatterplot(x=ajustados, y=residuos, ax=axes[1]); axes[1].axhline(0, color='red'); axes[1].set_title("2. Homocedasticidad")
    axes[2].plot(range(len(residuos)), residuos, marker='o'); axes[2].set_title("3. Independencia")
    sns.boxplot(x=factores[0], y=residuos, data=df, ax=axes[3]); axes[3].set_title("4. Aditividad")
    st.pyplot(fig)

    # Informe de Auditoría
    met1, met2, met3, met4 = st.columns(4)
    met1.metric("Normalidad (p)", f"{p_shapiro:.4f}", delta="Pasa" if p_shapiro > 0.05 else "Falla", delta_color="normal" if p_shapiro > 0.05 else "inverse")
    met2.metric("Homocedasticidad (p)", f"{p_levene:.4f}", delta="Pasa" if p_levene > 0.05 else "Falla", delta_color="normal" if p_levene > 0.05 else "inverse")
    met3.metric("Independencia (DW)", f"{dw_stat:.2f}", delta="Óptimo" if 1.5 < dw_stat < 2.5 else "Riesgo")
    met4.metric("Aditividad (p)", f"{p_aditividad:.4f}", delta="Pasa" if p_aditividad > 0.05 else "Falla", delta_color="normal" if p_aditividad > 0.05 else "inverse")

    if p_shapiro > 0.05 and p_levene > 0.05 and p_aditividad > 0.05:
        st.success("**Dictamen:** Supuestos validados satisfactoriamente. Procediendo al Análisis de Varianza.")
    else:
        st.warning("**Aviso:** Se detectan debilidades en los supuestos. Los resultados deben tomarse como tendencias.")

def ejecutar_flujo_v6(df, diseño, factores, respuesta):
    # 1. AEDA (Primero se exploran los datos)
    seccion_aeda(df, factores[0], respuesta)
    st.divider()
    
    # 2. MODELADO E INFERENCIA
    if diseño == "Diseño Factorial":
        formula = f"Q('{respuesta}') ~ C(Q('{factores[0]}')) * C(Q('{factores[1]}'))"
    elif diseño == "Diseño de Bloques (DBCA)":
        formula = f"Q('{respuesta}') ~ C(Q('{factores[0]}')) + C(Q('{factores[1]}'))"
    else:
        formula = f"Q('{respuesta}') ~ C(Q('{factores[0]}'))"

    try:
        modelo = ols(formula, data=df).fit()
        
        # Auditoría
        realizar_auditoria_supuestos(df, respuesta, modelo, factores)
        st.divider()
        
        # ANAVA
        st.header(f"📊 Tabla de Análisis de Varianza (ANAVA) - {diseño}")
        tabla = sm.stats.anova_lm(modelo, typ=2)
        st.table(tabla)
        
        p_val = tabla.iloc[0, 3]
        st.subheader("📝 Conclusión Técnica")
        if p_val < 0.05:
            st.success(f"**Significancia detectada (p={p_val:.4f}):** Existen diferencias reales entre tratamientos.")
            # Tukey
            st.header("🔍 Comparaciones de Medias (Tukey HSD)")
            ph = sp.posthoc_tukey(df, val_col=respuesta, group_col=factores[0])
            st.dataframe(ph.style.background_gradient(cmap='YlGnBu'))
            
            medias = df.groupby(factores[0])[respuesta].mean().sort_values()
            st.write(f"**Análisis de Rangos:** El mejor desempeño es de **{medias.index[-1]}** ({medias.max():.2f}) y el menor es de **{medias.index[0]}** ({medias.min():.2f}).")
        else:
            st.info(f"**Sin significancia (p={p_val:.4f}):** No se rechaza H₀. Los tratamientos son equivalentes.")

    except Exception as e:
        st.error(f"Error en el proceso: {e}")

# --- UI PRINCIPAL ---
st.title("📊 Master DOE VHMG v6: Exploración e Inferencia")
cargar_imagen_investigador()

archivo = st.file_uploader("Cargue el archivo experimental", type=['csv', 'txt'])

if archivo:
    df = pd.read_csv(archivo, sep=None, engine='python')
    columnas = df.columns.tolist()
    st.sidebar.header("⚙️ Configuración")
    tipo = st.sidebar.selectbox("Diseño:", ["DCA", "DBCA", "Diseño Factorial"])
    y = st.sidebar.selectbox("Respuesta (Y):", df.select_dtypes(include=[np.number]).columns)
    
    if tipo == "DCA":
        fct = [st.sidebar.selectbox("Tratamiento:", columnas)]
    else:
        fct = [st.sidebar.selectbox("Factor A:", columnas), st.sidebar.selectbox("Factor B / Bloque:", columnas)]

    if st.sidebar.button("⚡ Ejecutar Análisis Integral"):
        ejecutar_flujo_v6(df, tipo, fct, y)

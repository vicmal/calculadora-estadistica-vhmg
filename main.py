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
st.set_page_config(page_title="Suite DOE VHMG Master v4", layout="wide")
sns.set_theme(style="whitegrid")

def cargar_imagen_investigador():
    id_investigador = random.randint(1, 1000)
    url = f"https://picsum.photos/id/{id_investigador}/800/400"
    st.image(url, caption="Ingeniería y Ciencia de Datos - Ref. Autoría: Ing. Víctor Hugo Malavé Girón", use_container_width=True)

def redactar_conclusion_pro(p_valor, factor_nombre, alfa=0.05):
    st.subheader("📝 Dictamen Final del Ensayo")
    if p_valor < alfa:
        st.success(f"""
        **Resultado:** Estadísticamente Significativo (p = {p_valor:.4f} < {alfa}).
        
        **Interpretación del p-value:** La probabilidad de que las diferencias observadas se deban al azar es menor al 5%. 
        Por tanto, se **rechaza la Hipótesis Nula (H₀)** que planteaba igualdad entre tratamientos. 
        **Implicación:** El factor '{factor_nombre}' influye de manera determinante en la variable respuesta.
        """)
    else:
        st.info(f"""
        **Resultado:** No Significativo (p = {p_valor:.4f} > {alfa}).
        
        **Interpretación del p-value:** No existe evidencia suficiente para descartar que las diferencias sean fruto de la variabilidad natural o error experimental. 
        Se **acepta la Hipótesis Nula (H₀)**.
        **Implicación:** No se recomienda realizar cambios basados en el factor '{factor_nombre}', ya que su efecto no es consistente.
        """)

def analizar_post_hoc(df, factor, respuesta):
    st.subheader("🔍 Prueba de Comparación de Medias (Tukey HSD)")
    # Ejecutar prueba
    ph = sp.posthoc_tukey(df, val_col=respuesta, group_col=factor)
    st.dataframe(ph.style.background_gradient(cmap='YlGnBu'))
    
    # Identificar extremos
    medias = df.groupby(factor)[respuesta].mean().sort_values()
    mejor_t = medias.index[-1]
    peor_t = medias.index[0]
    
    st.markdown(f"""
    **Interpretación de Rangos:**
    * El tratamiento que presenta el **mayor promedio** es **{mejor_t}** con una media de `{medias.max():.2f}`.
    * El tratamiento con el **menor desempeño** es **{peor_t}** con una media de `{medias.min():.2f}`.
    * *Nota:* En la matriz superior, los valores p < 0.05 indican parejas de tratamientos que son significativamente diferentes entre sí.
    """)

def ejecutar_motor_v4(df, diseño, factores, respuesta):
    st.divider()
    # Construcción de fórmula (Lógica igual a v3)
    if diseño == "Diseño Completamente Aleatorizado (DCA)":
        formula = f"Q('{respuesta}') ~ C(Q('{factores[0]}'))"
    elif diseño == "Diseño de Bloques al Azar (DBCA)":
        formula = f"Q('{respuesta}') ~ C(Q('{factores[0]}')) + C(Q('{factores[1]}'))"
    elif diseño == "Diseño Factorial":
        formula = f"Q('{respuesta}') ~ C(Q('{factores[0]}')) * C(Q('{factores[1]}'))"
    else:
        terminos = " + ".join([f"C(Q('{f}'))" for f in factores])
        formula = f"Q('{respuesta}') ~ {terminos}"

    try:
        modelo = ols(formula, data=df).fit()
        df['Residuos'] = modelo.resid
        df['Ajustados'] = modelo.fittedvalues
        df['Orden'] = range(1, len(df) + 1)
        
        # 4 Supuestos
        st.header("🔬 Validación de los 4 Supuestos")
        fig, axes = plt.subplots(1, 4, figsize=(20, 4))
        sm.qqplot(df['Residuos'], line='s', ax=axes[0]); axes[0].set_title("1. Normalidad")
        sns.scatterplot(x=df['Ajustados'], y=df['Residuos'], ax=axes[1]); axes[1].axhline(0, color='red'); axes[1].set_title("2. Homocedasticidad")
        axes[2].plot(df['Orden'], df['Residuos'], marker='o'); axes[2].set_title("3. Independencia")
        sns.boxplot(x=factores[0], y='Residuos', data=df, ax=axes[3]); axes[3].set_title("4. Aditividad")
        st.pyplot(fig)

        # ANOVA e Interpretación
        st.header(f"📊 Resultados del {diseño}")
        tabla_anova = sm.stats.anova_lm(modelo, typ=2)
        st.table(tabla_anova)
        
        p_val_principal = tabla_anova.iloc[0, 3]
        redactar_conclusion_pro(p_val_principal, factores[0])
        
        if p_val_principal < 0.05:
            analizar_post_hoc(df, factores[0], respuesta)

    except Exception as e:
        st.error(f"Error técnico: {e}")

# --- INTERFAZ ---
st.title("🚀 Suite DOE Master v4 - Ing. Víctor Hugo Malavé Girón")
cargar_imagen_investigador()

archivo = st.file_uploader("Cargue su base de datos para análisis", type=['csv', 'txt'])

if archivo:
    df = pd.read_csv(archivo, sep=None, engine='python')
    columnas = df.columns.tolist()
    
    st.sidebar.header("⚙️ Configuración")
    tipo_diseño = st.sidebar.selectbox("Tipo de Diseño:", [
        "Diseño Completamente Aleatorizado (DCA)",
        "Diseño de Bloques al Azar (DBCA)",
        "Diseño Factorial",
        "Diseño Cuadrado Latino (DCL)"
    ])
    
    col_resp = st.sidebar.selectbox("Variable Respuesta (Y):", df.select_dtypes(include=[np.number]).columns)
    
    # Configuración de factores según diseño
    if tipo_diseño == "Diseño Completamente Aleatorizado (DCA)":
        factores = [st.sidebar.selectbox("Factor Tratamiento:", columnas)]
    elif tipo_diseño in ["Diseño de Bloques al Azar (DBCA)", "Diseño Factorial"]:
        f1 = st.sidebar.selectbox("Factor Principal:", columnas)
        f2 = st.sidebar.selectbox("Factor Secundario/Bloque:", columnas)
        factores = [f1, f2]
    else:
        factores = st.sidebar.multiselect("Seleccione Factores:", columnas)

    if st.sidebar.button("⚡ Ejecutar Análisis Profesional"):
        ejecutar_motor_v4(df, tipo_diseño, factores, col_resp)

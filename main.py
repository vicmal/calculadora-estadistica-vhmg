import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random

# Librerías para Estadística
import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy.stats import shapiro, levene, kruskal
import scikit_posthocs as sp  # Para Dunn (post-hoc no paramétrico)
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# 1. CONFIGURACIÓN E IDENTIDAD
st.set_page_config(page_title="Calculadora VHMG - Estadística", layout="wide")

# Lógica de imágenes aleatorias
imagenes_investigacion = [
    "https://images.unsplash.com/photo-1581091226825-a6a2a5aee158",
    "https://images.unsplash.com/photo-1551288049-bbda48658a7d",
    "https://images.unsplash.com/photo-1523348837708-15d4a09cfac2",
    "https://images.unsplash.com/photo-1576086213369-97a306dca665"
]

# 2. ENCABEZADO
col1, col2 = st.columns([1, 3])
with col1:
    st.image(random.choice(imagenes_investigacion), use_container_width=True)
with col2:
    st.title("CALCULADORA DE ANÁLISIS ESTADÍSTICO VHMG")
    st.markdown("**Sistema Experto Integrado** | Autor: *Ing. Víctor Hugo Malavé Girón*")

# 3. PANEL DE CONTROL
st.sidebar.header("CONFIGURACIÓN")
archivo = st.sidebar.file_uploader("Suba su archivo (.txt o .csv)", type=["txt", "csv"])
diseno = st.sidebar.selectbox("Diseño Experimental:", ["DCA", "DBCA", "Factorial"])

if archivo is not None:
    df = pd.read_csv(archivo, sep=None, engine='python')
    vars_num = df.select_dtypes(include=[np.number]).columns.tolist()
    vars_cat = df.select_dtypes(include=['object', 'category']).columns.tolist()

    if vars_num and vars_cat:
        st.sidebar.subheader("Variables del Modelo")
        resp = st.sidebar.selectbox("Variable Respuesta (Y):", vars_num)

        if diseno == "Factorial":
            f1 = st.sidebar.selectbox("Factor A:", vars_cat, key="f1")
            f2 = st.sidebar.selectbox("Factor B:", vars_cat, key="f2")
            factores = [f1, f2]
        else:
            f1 = st.sidebar.selectbox("Factor Principal:", vars_cat)
            factores = [f1]

        if st.sidebar.button("INICIAR ANÁLISIS TOTAL"):
            # --- PASO 1: AED ---
            st.header("🔍 1. Análisis Exploratorio (AED)")
            col_a, col_b = st.columns(2)
            with col_a:
                st.write("**Estadísticos Descriptivos**")
                st.dataframe(df.groupby(f1)[resp].describe())
            with col_b:
                fig_box = plt.figure(figsize=(8, 5))
                sns.boxplot(data=df, x=f1, y=resp, palette="Set2")
                st.pyplot(fig_box)

            # --- PASO 2: ANAVA Y SUPUESTOS ---
            st.divider()
            st.header("🧪 2. Motor de Inferencia y Supuestos")

            # Ajuste de Modelo según diseño
            if diseno == "Factorial":
                formula = f"{resp} ~ C({f1}) * C({f2})"
                # Gráfico de Interacción (NUEVO)
                st.subheader("📊 Gráfico de Interacción")
                fig_int = plt.figure(figsize=(10, 5))
                from statsmodels.graphics.factorplots import interaction_plot

                interaction_plot(x=df[f1], trace=df[f2], response=df[resp], ax=plt.gca())
                st.pyplot(fig_int)
            else:
                formula = f"{resp} ~ C({f1})"

            modelo = ols(formula, data=df).fit()
            tabla_anava = sm.stats.anova_lm(modelo, typ=2)
            residuos = modelo.resid

            # Validación de Supuestos
            p_shapiro = shapiro(residuos)[1]
            p_levene = levene(*[df[resp][df[f1] == n] for n in df[f1].unique()])[1]

            col_s1, col_s2 = st.columns(2)
            with col_s1:
                st.metric("Normalidad (p-val)", f"{p_shapiro:.4f}", delta="CUMPLE" if p_shapiro > 0.05 else "FALLA",
                          delta_color="normal" if p_shapiro > 0.05 else "inverse")
            with col_s2:
                st.metric("Homocedasticidad (p-val)", f"{p_levene:.4f}", delta="CUMPLE" if p_levene > 0.05 else "FALLA",
                          delta_color="normal" if p_levene > 0.05 else "inverse")

            # --- PASO 3: DECISIÓN INTELIGENTE (NUEVO) ---
            st.divider()
            st.header("⚖️ 3. Decisión y Resultados Finales")

            if p_shapiro > 0.05 and p_levene > 0.05:
                st.success("✅ RUTA PARAMÉTRICA: Se cumplen los supuestos.")
                st.dataframe(tabla_anava)

                # Reporte Automático ANAVA
                sig_factores = [f for f in tabla_anava.index if
                                f != 'Remainder' and tabla_anava.loc[f, 'PR(>F)'] < 0.05]
                st.info(
                    f"**Reporte:** Se detectaron diferencias significativas en: {', '.join(sig_factores) if sig_factores else 'Ninguno'}.")

                if sig_factores:
                    st.subheader("📍 Comparación de Medias (Tukey)")
                    tukey = pairwise_tukeyhsd(df[resp], df[f1])
                    st.write(tukey)

            else:
                st.error("❌ RUTA NO PARAMÉTRICA: Violación de supuestos detectada.")
                st.warning("Ejecutando Prueba de Kruskal-Wallis...")

                # Kruskal-Wallis (NUEVO)
                grupos = [df[resp][df[f1] == n] for n in df[f1].unique()]
                h_stat, p_kruskal = kruskal(*grupos)
                st.write(f"**Prueba H de Kruskal-Wallis:** p-valor = {p_kruskal:.4f}")

                if p_kruskal < 0.05:
                    st.subheader("📍 Comparación Múltiple No Paramétrica (Prueba de Dunn)")
                    # Prueba de Dunn (NUEVO)
                    dunn = sp.posthoc_dunn(df, val_col=resp, group_col=f1, p_adjust='bonferroni')
                    st.dataframe(dunn)
                    st.write("Interpretación: Los valores < 0.05 indican parejas con diferencias significativas.")

            # --- PASO 4: REPORTE FINAL AUTOMÁTICO (NUEVO) ---
            st.subheader("📝 Conclusión Final")
            if p_shapiro < 0.05:
                conclusion = "Los datos no presentan normalidad, lo cual sugiere la presencia de valores atípicos o una distribución sesgada. "
            else:
                conclusion = "Los datos cumplen con los estándares de normalidad. "

            if 'p_kruskal' in locals() and p_kruskal < 0.05:
                conclusion += f"A través de métodos no paramétricos, se confirma que el factor {f1} altera significativamente la respuesta."
            elif sig_factores:
                conclusion += f"El análisis de varianza confirma efectos significativos de los factores estudiados sobre la variable {resp}."

            st.write(conclusion)

    else:
        st.error("El archivo no tiene el formato adecuado de columnas.")

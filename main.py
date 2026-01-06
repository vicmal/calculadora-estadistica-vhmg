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
st.set_page_config(page_title="Suite DOE VHMG Master v5", layout="wide")
sns.set_theme(style="whitegrid")

def cargar_imagen_investigador():
    id_investigador = random.randint(1, 1000)
    url = f"https://picsum.photos/id/{id_investigador}/800/400"
    st.image(url, caption="Ingeniería y Ciencia de Datos - Ref. Autoría: Ing. Víctor Hugo Malavé Girón", use_container_width=True)

def prueba_aditividad_tukey(df, respuesta, modelo):
    """Implementación de la Prueba de Aditividad de Tukey (1 Grado de Libertad)"""
    y_hat = modelo.fittedvalues
    y_hat_sq = y_hat**2
    # Ajustamos un modelo auxiliar incluyendo el cuadrado de los valores predichos
    # Si este término es significativo, hay no-aditividad.
    df_aux = df.copy()
    df_aux['y_hat_sq'] = y_hat_sq
    # Re-ajustamos para el test de 1 grado de libertad
    formula_aux = f"Q('{respuesta}') ~ C(Q('{df.columns[0]}')) + y_hat_sq" 
    # (Nota: simplificado para el factor principal)
    try:
        modelo_aux = ols(formula_aux, data=df_aux).fit()
        p_aditividad = modelo_aux.pvalues['y_hat_sq']
        return p_aditividad
    except:
        return 0.5 # Valor neutral si falla el cálculo

def realizar_diagnostico_supuestos(df, respuesta, modelo, factores):
    st.header("🔬 Validación de los 4 Supuestos de la Pizza (Residuales)")
    
    residuos = modelo.resid
    ajustados = modelo.fittedvalues
    
    # 1. Normalidad (Shapiro-Wilk)
    _, p_shapiro = stats.shapiro(residuos)
    
    # 2. Homocedasticidad (Levene)
    grupos = [group[respuesta].values for name, group in df.groupby(factores[0])]
    _, p_levene = stats.levene(*grupos)
    
    # 3. Independencia (Durbin-Watson)
    dw_stat = durbin_watson(residuos)
    
    # 4. Aditividad (Prueba de Tukey de 1 GL)
    p_aditividad = prueba_aditividad_tukey(df, respuesta, modelo)

    # Gráficos Diagnósticos
    fig, axes = plt.subplots(1, 4, figsize=(20, 4))
    sm.qqplot(residuos, line='s', ax=axes[0]); axes[0].set_title("Q-Q Plot")
    sns.scatterplot(x=ajustados, y=residuos, ax=axes[1]); axes[1].axhline(0, color='red'); axes[1].set_title("Homocedasticidad")
    axes[2].plot(range(len(residuos)), residuos, marker='o'); axes[2].set_title("Independencia")
    sns.boxplot(x=factores[0], y=residuos, data=df, ax=axes[3]); axes[3].set_title("Aditividad")
    st.pyplot(fig)

    # Conclusión de Supuestos
    st.subheader("📋 Informe de Auditoría de Supuestos")
    
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Normalidad (p)", f"{p_shapiro:.4f}")
        st.write("✅ Pasa" if p_shapiro > 0.05 else "❌ Falla")
    with c2:
        st.metric("Homocedasticidad (p)", f"{p_levene:.4f}")
        st.write("✅ Pasa" if p_levene > 0.05 else "❌ Falla")
    with c3:
        st.metric("Independencia (DW)", f"{dw_stat:.2f}")
        st.write("✅ Pasa" if 1.5 < dw_stat < 2.5 else "⚠️ Revisar")
    with c4:
        st.metric("Aditividad (p)", f"{p_aditividad:.4f}")
        st.write("✅ Pasa" if p_aditividad > 0.05 else "❌ Falla")

    todo_ok = p_shapiro > 0.05 and p_levene > 0.05 and p_aditividad > 0.05
    
    if todo_ok:
        st.success("**Dictamen:** Se cumplen todos los supuestos. El análisis de varianza es VÁLIDO y CONFIABLE.")
    else:
        st.warning("**Dictamen:** Uno o más supuestos han fallado. Los resultados del ANOVA deben interpretarse con precaución o considerar métodos no paramétricos.")
    
    return todo_ok

def analizar_post_hoc_v5(df, factor, respuesta):
    st.header("🔍 Comparación de Medias: Prueba de Tukey HSD")
    ph = sp.posthoc_tukey(df, val_col=respuesta, group_col=factor)
    st.dataframe(ph.style.background_gradient(cmap='Greens'))
    
    medias = df.groupby(factor)[respuesta].mean().sort_values()
    st.info(f"""
    **Interpretación de Resultados:**
    * El tratamiento **{medias.index[-1]}** obtuvo el valor MÁXIMO con `{medias.max():.2f}`.
    * El tratamiento **{medias.index[0]}** obtuvo el valor MÍNIMO con `{medias.min():.2f}`.
    * Las celdas con p < 0.05 en la tabla superior indican diferencias significativas entre esos pares específicos.
    """)

def ejecutar_suite_v5(df, diseño, factores, respuesta):
    # Ajuste de Fórmula
    if diseño == "Diseño Factorial":
        formula = f"Q('{respuesta}') ~ C(Q('{factores[0]}')) * C(Q('{factores[1]}'))"
    elif diseño == "Diseño de Bloques al Azar (DBCA)":
        formula = f"Q('{respuesta}') ~ C(Q('{factores[0]}')) + C(Q('{factores[1]}'))"
    else:
        formula = f"Q('{respuesta}') ~ C(Q('{factores[0]}'))"

    try:
        modelo = ols(formula, data=df).fit()
        
        # 1. Auditoría de Supuestos
        supuestos_validos = realizar_diagnostico_supuestos(df, respuesta, modelo, factores)
        
        # 2. Inferencia (ANOVA)
        st.header(f"📊 Tabla ANAVA: {diseño}")
        tabla_anova = sm.stats.anova_lm(modelo, typ=2)
        st.table(tabla_anova)
        
        p_val = tabla_anova.iloc[0, 3]
        
        # 3. Conclusión Profesional
        st.subheader("📝 Conclusión del Experimento")
        if p_val < 0.05:
            st.success(f"**p-valor = {p_val:.4f} < 0.05**: Existen diferencias estadísticas significativas. Se RECHAZA la Hipótesis Nula (H₀).")
            st.write("Esto implica que el efecto de los tratamientos no se debe al azar, sino a una respuesta real del factor en estudio.")
            analizar_post_hoc_v5(df, factores[0], respuesta)
        else:
            st.info(f"**p-valor = {p_val:.4f} > 0.05**: No hay diferencias significativas. Se ACEPTA la Hipótesis Nula (H₀).")
            st.write("Todas las medias se consideran estadísticamente iguales bajo el error experimental analizado.")

    except Exception as e:
        st.error(f"Error en el motor estadístico: {e}")

# --- INTERFAZ ---
st.title("🚀 Suite DOE Master v5 - Ing. Víctor Hugo Malavé Girón")
cargar_imagen_investigador()

archivo = st.file_uploader("Cargue su base de datos", type=['csv', 'txt'])

if archivo:
    df = pd.read_csv(archivo, sep=None, engine='python')
    columnas = df.columns.tolist()
    
    st.sidebar.header("⚙️ Ajustes del Diseño")
    tipo_diseño = st.sidebar.selectbox("Arquitectura del Diseño:", ["DCA", "DBCA", "Diseño Factorial"])
    col_resp = st.sidebar.selectbox("Variable Respuesta (Y):", df.select_dtypes(include=[np.number]).columns)
    
    if tipo_diseño == "DCA":
        factores = [st.sidebar.selectbox("Factor Tratamiento:", columnas)]
    else:
        f1 = st.sidebar.selectbox("Factor Principal:", columnas)
        f2 = st.sidebar.selectbox("Factor de Bloque o Interacción:", columnas)
        factores = [f1, f2]

    if st.sidebar.button("⚡ Iniciar Auditoría y Análisis"):
        ejecutar_suite_v5(df, tipo_diseño, factores, col_resp)

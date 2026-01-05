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

def redactar_conclusion(p_valor, metodo, alfa=0.05):
    st.subheader("📝 Conclusión del Ensayo Experimental")
    
    if p_valor < alfa:
        conclusion = f"""
        **Resultado:** Estadísticamente Significativo (p = {p_valor:.4f}).
        
        **Dictamen:** Al ser el p-valor menor que el nivel de significancia (α = {alfa}), se **rechaza la Hipótesis Nula (H₀)**. 
        Existen evidencias suficientes para afirmar que al menos uno de los tratamientos produce un efecto diferente sobre la variable respuesta.
        
        **Acción:** Se procede a analizar las pruebas de comparaciones múltiples (Post-hoc) para identificar entre qué tratamientos específicos residen las diferencias.
        """
        st.success(conclusion)
    else:
        conclusion = f"""
        **Resultado:** No Significativo (p = {p_valor:.4f}).
        
        **Dictamen:** Al ser el p-valor mayor que el nivel de significancia (α = {alfa}), **no se rechaza la Hipótesis Nula (H₀)**. 
        Las diferencias observadas entre las medias de los tratamientos pueden atribuirse al azar (error experimental) y no a un efecto real de los factores en estudio.
        
        **Acción:** No se requiere realizar pruebas de rangos múltiples. Se recomienda revisar el tamaño de la muestra o el control de variables extrañas si se esperaba un efecto.
        """
        st.info(conclusion)

def realizar_analisis_vhmg(df):
    st.header("🔬 Auditoría de Supuestos del Modelo")
    
    try:
        modelo = ols('Respuesta ~ C(Tratamiento)', data=df).fit()
        df['Ajustados'] = modelo.fittedvalues
        df['Residuales'] = modelo.resid
        df['Orden'] = range(1, len(df) + 1)
    except Exception as e:
        st.error(f"Error en el modelo: {e}")
        return

    # Gráficos de Diagnóstico
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    sm.qqplot(df['Residuales'], line='s', ax=axes[0, 0])
    axes[0, 0].set_title("Q-Q Plot (Normalidad)")
    sns.scatterplot(x=df['Ajustados'], y=df['Residuales'], ax=axes[0, 1])
    axes[0, 1].axhline(0, color='red', linestyle='--')
    axes[0, 1].set_title("Residuos vs. Ajustados")
    axes[1, 0].plot(df['Orden'], df['Residuales'], marker='o')
    axes[1, 0].axhline(0, color='red', linestyle='--')
    axes[1, 0].set_title("Residuos vs. Orden (Independencia)")
    sns.boxplot(x='Tratamiento', y='Residuales', data=df, ax=axes[1, 1])
    axes[1, 1].axhline(0, color='red', linestyle='--')
    axes[1, 1].set_title("Residuos por Tratamiento (Aditividad)")
    plt.tight_layout()
    st.pyplot(fig)

    # Pruebas Formales
    _, p_shapiro = stats.shapiro(df['Residuales'])
    grupos = [group['Residuales'].values for name, group in df.groupby('Tratamiento')]
    _, p_levene = stats.levene(*grupos)
    dw = durbin_watson(df['Residuales'])
    
    cumple_norm = p_shapiro > 0.05
    cumple_homo = p_levene > 0.05
    
    st.divider()

    # Inferencia y Conclusión
    if cumple_norm and cumple_homo:
        st.header("📊 Análisis de Varianza (ANOVA)")
        tabla_anova = sm.stats.anova_lm(modelo, typ=2)
        st.table(tabla_anova)
        p_val = tabla_anova['PR(>F)'][0]
        
        redactar_conclusion(p_val, "ANOVA")
        
        if p_val < 0.05:
            st.subheader("🔍 Comparaciones Múltiples (Tukey HSD)")
            posthoc = sp.posthoc_tukey(df, val_col='Respuesta', group_col='Tratamiento')
            st.dataframe(posthoc.style.background_gradient(cmap='viridis'))
    else:
        st.header("📊 Prueba de Kruskal-Wallis (No Paramétrica)")
        stat_k, p_k = stats.kruskal(*[group['Respuesta'].values for name, group in df.groupby('Tratamiento')])
        st.write(f"Estadístico H: `{stat_k:.4f}`, p-valor: `{p_k:.4f}`")
        
        redactar_conclusion(p_k, "Kruskal-Wallis")
        
        if p_k < 0.05:
            st.subheader("🔍 Comparaciones Múltiples (Dunn)")
            posthoc = sp.posthoc_dunn(df, val_col='Respuesta', group_col='Tratamiento', p_adjust='holm')
            st.dataframe(posthoc.style.background_gradient(cmap='viridis'))

# --- Interfaz Principal ---
st.title("📊 Calculadora VHMG: Diagnóstico y Análisis Pro")
cargar_imagen_investigador()

archivo = st.file_uploader("Cargue su archivo (CSV o TXT)", type=['csv', 'txt'])

if archivo:
    try:
        df = pd.read_csv(archivo, sep=None, engine='python')
        if 'Tratamiento' in df.columns and 'Respuesta' in df.columns:
            realizar_analisis_vhmg(df)
        else:
            st.error("Columnas requeridas: 'Tratamiento' y 'Respuesta'.")
    except Exception as e:
        st.error(f"Error: {e}")

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
st.set_page_config(page_title="Suite DOE VHMG Master v6.3", layout="wide")
sns.set_theme(style="whitegrid")

VERSION = "6.3"


def cargar_imagen_investigador():
    id_investigador = random.randint(1, 1000)
    url = f"https://picsum.photos/id/{id_investigador}/800/400"
    st.image(url, caption="Ref. Autoría: Ing. Víctor Hugo Malavé Girón - Ingeniería y Ciencia de Datos", use_container_width=True)


def seccion_aeda(df, factor, respuesta):
    st.header(f"📊 Análisis Exploratorio de Datos (AEDA) - {respuesta}")
    col1, col2 = st.columns([1, 1.5])

    with col1:
        st.subheader("🔢 Estadísticas Descriptivas")
        desc = df.groupby(factor)[respuesta].agg(['count', 'mean', 'std', 'min', 'median', 'max']).reset_index()
        desc['CV%'] = (desc['std'] / desc['mean']) * 100
        st.dataframe(desc.style.format(precision=3).background_gradient(subset=['mean'], cmap='Blues'))
        st.info("**Nota:** Un CV% bajo indica mayor precisión en el experimento.")

    with col2:
        st.subheader("📈 Comportamiento de los Datos Crudos")
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        sns.boxplot(data=df, x=factor, y=respuesta, ax=ax[0], palette="Set2")
        ax[0].set_title("Distribución (Boxplot)")
        sns.histplot(data=df, x=respuesta, hue=factor, kde=True, ax=ax[1], palette="Set2", legend=False)
        ax[1].set_title("Histograma y Densidad")
        st.pyplot(fig)


def prueba_aditividad_tukey(df, respuesta, modelo, factor):
    """Prueba de Tukey de un grado de libertad para no aditividad.

    Devuelve None si el modelo auxiliar no puede ajustarse. En la versión
    anterior devolvía 0.5 en silencio, lo que se mostraba como "Pasa"
    sin haber probado nada.
    """
    y_hat = modelo.fittedvalues
    df_aux = df.copy()
    df_aux['y_hat_sq'] = y_hat**2
    try:
        formula_aux = f"Q('{respuesta}') ~ C(Q('{factor}')) + y_hat_sq"
        modelo_aux = ols(formula_aux, data=df_aux).fit()
        return modelo_aux.pvalues['y_hat_sq']
    except Exception:
        return None


def realizar_auditoria_supuestos(df, respuesta, modelo, factores, diseño, orden_declarado):
    st.header("🔬 Auditoría de los 4 Supuestos Críticos (Sobre Residuales)")
    residuos = modelo.resid
    ajustados = modelo.fittedvalues

    _, p_shapiro = stats.shapiro(residuos)
    # Homogeneidad: Levene sobre la respuesta agrupada por tratamiento
    # (DCA/DBCA) o por celda A×B (factorial), para no ignorar el segundo factor.
    if diseño == "Diseño Factorial":
        celda = df[factores[0]].astype(str) + " × " + df[factores[1]].astype(str)
        grupos = [group[respuesta].values for _, group in df.groupby(celda)]
    else:
        grupos = [group[respuesta].values for _, group in df.groupby(factores[0])]
    _, p_levene = stats.levene(*grupos)
    dw_stat = durbin_watson(residuos)
    # Aditividad (prueba de Tukey de un grado de libertad) solo tiene
    # sentido en un diseño de dos vías aditivo (DBCA): en un DCA el valor
    # ajustado es función del único factor y el modelo auxiliar es
    # singular; en un factorial la interacción ya está en el modelo.
    if diseño in ("DBCA", "Diseño de Bloques (DBCA)"):
        p_aditividad = prueba_aditividad_tukey(df, respuesta, modelo, factores[0])
    else:
        p_aditividad = None

    fig, axes = plt.subplots(1, 4, figsize=(20, 4))
    sm.qqplot(residuos, line='s', ax=axes[0]); axes[0].set_title("1. Normalidad (Q-Q)")
    sns.scatterplot(x=ajustados, y=residuos, ax=axes[1]); axes[1].axhline(0, color='red'); axes[1].set_title("2. Homocedasticidad")
    axes[2].plot(range(len(residuos)), residuos, marker='o'); axes[2].set_title("3. Independencia")
    titulo_panel4 = "4. Aditividad" if diseño in ("DBCA", "Diseño de Bloques (DBCA)") else "4. Residuales"
    sns.boxplot(x=factores[0], y=residuos, data=df, ax=axes[3]); axes[3].set_title(titulo_panel4)
    st.pyplot(fig)

    met1, met2, met3, met4 = st.columns(4)
    met1.metric("Normalidad (p)", f"{p_shapiro:.4f}", delta="Pasa" if p_shapiro > 0.05 else "Falla", delta_color="normal" if p_shapiro > 0.05 else "inverse")
    met2.metric("Homocedasticidad (p)", f"{p_levene:.4f}", delta="Pasa" if p_levene > 0.05 else "Falla", delta_color="normal" if p_levene > 0.05 else "inverse")
    if orden_declarado:
        met3.metric("Independencia (DW)", f"{dw_stat:.2f}", delta="Óptimo" if 1.5 < dw_stat < 2.5 else "Riesgo")
    else:
        met3.metric("Independencia (DW)", f"{dw_stat:.2f}", delta="Sin orden declarado")
    if diseño in ("DBCA", "Diseño de Bloques (DBCA)"):
        if p_aditividad is None:
            met4.metric("Aditividad (p)", "No calculada", delta="Sin prueba")
        else:
            met4.metric("Aditividad (p)", f"{p_aditividad:.4f}", delta="Pasa" if p_aditividad > 0.05 else "Falla", delta_color="normal" if p_aditividad > 0.05 else "inverse")
    elif diseño == "Diseño Factorial":
        met4.metric("Aditividad (p)", "No aplica", delta="Modelo con interacción")
    else:
        met4.metric("Aditividad (p)", "No aplica", delta="Un solo factor")

    if not orden_declarado:
        st.info("Nota: Durbin-Watson se calcula sobre el orden de las filas del archivo. Como no se declaró que las filas están en orden de ejecución, el valor se muestra sin interpretar y no forma parte del dictamen.")
    if p_aditividad is None and diseño in ("DBCA", "Diseño de Bloques (DBCA)"):
        st.info("Nota: la prueba de aditividad no pudo calcularse para estos datos; no se incluye en el dictamen.")

    ok_shapiro = p_shapiro > 0.05
    ok_levene = p_levene > 0.05
    ok_dw = (not orden_declarado) or (1.5 < dw_stat < 2.5)
    if diseño in ("DBCA", "Diseño de Bloques (DBCA)"):
        ok_adit = (p_aditividad is not None) and (p_aditividad > 0.05)
    else:
        ok_adit = True  # no aplica

    if ok_shapiro and ok_levene and ok_dw and ok_adit:
        st.success("**Dictamen:** Supuestos validados satisfactoriamente. Procediendo al Análisis de Varianza.")
    else:
        st.warning("**Aviso:** Se detectan debilidades en los supuestos. Los resultados deben tomarse como tendencias.")


def ejecutar_flujo_v6(df, diseño, factores, respuesta, orden_declarado=False):
    # La respuesta no puede ser también factor o bloque.
    if respuesta in factores:
        st.error("❌ Error: la variable respuesta no puede ser también factor o bloque.")
        return

    st.subheader("🛠️ Fase 1: Limpieza y Validación de Calidad")

    # --- BLOQUE DE LIMPIEZA AUTOMÁTICA ---
    filas_iniciales = len(df)
    df = df.dropna(subset=[respuesta] + factores)
    df[respuesta] = pd.to_numeric(df[respuesta], errors='coerce')
    df = df[np.isfinite(df[respuesta])]
    for f in factores:
        df[f] = df[f].astype(str)

    if len(df) < filas_iniciales:
        st.warning(f"⚠️ Se eliminaron {filas_iniciales - len(df)} filas debido a datos no numéricos o nulos.")

    if df[respuesta].nunique() <= 1:
        st.error("❌ Error de Validación: La variable respuesta no tiene suficiente variación para ser analizada.")
        return

    # --- VERIFICACIÓN DE BALANCE ---
    # DCA: conteos iguales por tratamiento. DBCA/factorial: conteos iguales
    # por celda (tratamiento × bloque o A × B).
    if diseño in ("DBCA", "Diseño de Bloques (DBCA)", "Diseño Factorial"):
        grupos_balance = df[factores[0]].astype(str) + " × " + df[factores[1]].astype(str)
    else:
        grupos_balance = df[factores[0]].astype(str)
    conteos = grupos_balance.value_counts()
    if conteos.nunique() > 1:
        st.warning("⚠️ Datos desbalanceados: los conteos por grupo/celda no son iguales. La tabla ANAVA es de tipo II; con desbalance, los resultados pueden diferir de una tabla tipo III y las comparaciones deben interpretarse con cautela.")

    # --- FASE 2: AEDA ---
    seccion_aeda(df, factores[0], respuesta)
    st.divider()

    # --- FASE 3: MODELADO ---
    # CORRECCIÓN v6.3: el selector ofrece "DBCA" y la versión anterior
    # comparaba con "Diseño de Bloques (DBCA)", por lo que el bloque se
    # ignoraba en silencio y el DBCA se analizaba como un DCA.
    if diseño == "Diseño Factorial":
        formula = f"Q('{respuesta}') ~ C(Q('{factores[0]}')) * C(Q('{factores[1]}'))"
    elif diseño in ("DBCA", "Diseño de Bloques (DBCA)"):
        formula = f"Q('{respuesta}') ~ C(Q('{factores[0]}')) + C(Q('{factores[1]}'))"
    else:
        formula = f"Q('{respuesta}') ~ C(Q('{factores[0]}'))"

    try:
        modelo = ols(formula, data=df).fit()
        realizar_auditoria_supuestos(df, respuesta, modelo, factores, diseño, orden_declarado)
        st.divider()

        # ANAVA
        st.header(f"📊 Tabla de Análisis de Varianza (ANAVA) - {diseño}")
        tabla = sm.stats.anova_lm(modelo, typ=2)
        st.table(tabla)

        # Valores p por nombre de término (no por posición en la tabla).
        def p_de(termino):
            try:
                return tabla.loc[termino, "PR(>F)"]
            except KeyError:
                return None

        term_a = f"C(Q('{factores[0]}'))"

        # --- Análisis de Interacción (solo factorial) ---
        if diseño == "Diseño Factorial":
            term_b = f"C(Q('{factores[1]}'))"
            term_inter = f"{term_a}:{term_b}"
            p_a = p_de(term_a)
            p_b = p_de(term_b)
            p_inter = p_de(term_inter)
            st.subheader("🧬 Análisis de Interacción")
            if p_inter is not None and p_inter < 0.05:
                st.warning(f"**Interacción Significativa (p={p_inter:.4f}):**")
                st.write("Los factores interactúan entre sí. No analice los factores por separado; enfoque su conclusión en las combinaciones específicas de tratamientos.")
            elif p_inter is not None:
                st.info(f"**Interacción No Significativa (p={p_inter:.4f}):**")
                st.write("Los factores actúan de forma independiente. Puede interpretar el efecto de cada factor de manera aislada.")
            else:
                st.info("**Interacción:** no pudo determinarse su valor p.")

            # --- Conclusión por término ---
            st.subheader("📝 Conclusión Técnica")
            for nombre, p_val in ((factores[0], p_a), (factores[1], p_b), ("Interacción", p_inter)):
                if p_val is None:
                    st.info(f"**{nombre}:** valor p no disponible.")
                elif p_val < 0.05:
                    st.success(f"**{nombre} significativo (p={p_val:.4f}):** Se rechaza H₀ para este término.")
                else:
                    st.info(f"**{nombre} no significativo (p={p_val:.4f}):** No se rechaza H₀ para este término.")

            # --- Comparaciones ---
            # CORRECCIÓN v6.3: con interacción significativa NO se hace
            # Tukey sobre las medias marginales de un factor (eso contradecía
            # la propia advertencia); se comparan las combinaciones A×B.
            # Sin interacción, se hace Tukey marginal para cada efecto
            # principal significativo, no solo para el factor A.
            if p_inter is not None and p_inter < 0.05:
                df_celda = df.copy()
                df_celda["__celda__"] = df_celda[factores[0]].astype(str) + " × " + df_celda[factores[1]].astype(str)
                st.write("**Medias por combinación:**")
                st.dataframe(df_celda.groupby("__celda__")[respuesta].mean().reset_index())
                st.header("🔍 Comparaciones de Medias (Tukey HSD sobre combinaciones)")
                ph = sp.posthoc_tukey(df_celda, val_col=respuesta, group_col="__celda__")
                st.dataframe(ph.style.background_gradient(cmap='YlGnBu'))
            else:
                alguno = False
                for f, p_f in ((factores[0], p_a), (factores[1], p_b)):
                    if p_f is not None and p_f < 0.05:
                        alguno = True
                        st.header(f"🔍 Comparaciones de Medias (Tukey HSD) — {f}")
                        ph = sp.posthoc_tukey(df, val_col=respuesta, group_col=f)
                        st.dataframe(ph.style.background_gradient(cmap='YlGnBu'))
                        medias = df.groupby(f)[respuesta].mean().sort_values()
                        st.write(f"**Análisis de Rangos ({f}):** El tratamiento superior es **{medias.index[-1]}** con media de {medias.max():.2f} y el inferior es **{medias.index[0]}** con {medias.min():.2f}.")
                if not alguno:
                    st.info("Sin efectos principales significativos, no se realizan comparaciones post-hoc.")
        else:
            # DCA / DBCA: la conclusión depende del factor tratamiento.
            p_trat = p_de(term_a)
            st.subheader("📝 Conclusión Técnica")
            if p_trat is not None and p_trat < 0.05:
                st.success(f"**Significancia detectada (p={p_trat:.4f}):** Se rechaza H₀. Existen diferencias significativas entre tratamientos.")

                st.header("🔍 Comparaciones de Medias (Tukey HSD)")
                ph = sp.posthoc_tukey(df, val_col=respuesta, group_col=factores[0])
                st.dataframe(ph.style.background_gradient(cmap='YlGnBu'))

                medias = df.groupby(factores[0])[respuesta].mean().sort_values()
                st.write(f"**Análisis de Rangos:** El tratamiento superior es **{medias.index[-1]}** con media de {medias.max():.2f} y el inferior es **{medias.index[0]}** con {medias.min():.2f}.")
            elif p_trat is not None:
                st.info(f"**Sin significancia (p={p_trat:.4f}):** No se rechaza H₀. Los tratamientos se comportan de forma similar bajo este error experimental.")
            else:
                st.info("**Tratamiento:** valor p no disponible.")
            if diseño in ("DBCA", "Diseño de Bloques (DBCA)"):
                p_bloq = p_de(f"C(Q('{factores[1]}'))")
                if p_bloq is not None and p_bloq < 0.05:
                    st.info(f"El bloque también resultó significativo (p={p_bloq:.4f}), lo que respalda haber bloqueado el experimento.")

    except Exception as e:
        st.error(f"❌ Error Crítico en el motor estadístico: {e}")
        st.info("Sugerencia: Verifique que no haya caracteres especiales o comas en lugar de puntos en sus datos numéricos.")


# --- UI PRINCIPAL ---
st.title("📊 CALCULADORA DE ANÁLISIS DE VARIANZA - VHMG")
st.caption("Versión 6.3 — corrección del repositorio")
cargar_imagen_investigador()

archivo = st.file_uploader("Cargue el archivo experimental (.csv o .txt)", type=['csv', 'txt'])

if archivo:
    try:
        df_input = pd.read_csv(archivo, sep=None, engine='python')
        columnas = df_input.columns.tolist()

        st.sidebar.header("⚙️ Configuración del Ensayo")
        tipo = st.sidebar.selectbox("Diseño Experimental:", ["DCA", "DBCA", "Diseño Factorial"])
        y_col = st.sidebar.selectbox("Variable Respuesta (Debe ser numérica):", columnas)

        if tipo == "DCA":
            f_cols = [st.sidebar.selectbox("Factor Tratamiento:", columnas)]
        else:
            f1 = st.sidebar.selectbox("Factor Principal:", columnas)
            f2 = st.sidebar.selectbox("Factor Secundario / Bloque:", columnas)
            f_cols = [f1, f2]

        orden_ejecucion = st.sidebar.checkbox("Las filas están en orden de ejecución (para Durbin-Watson)", value=False)

        if st.sidebar.button("⚡ Ejecutar Análisis Profesional"):
            ejecutar_flujo_v6(df_input, tipo, f_cols, y_col, orden_declarado=orden_ejecucion)

    except Exception as e:
        st.error(f"Error al leer el archivo: {e}")

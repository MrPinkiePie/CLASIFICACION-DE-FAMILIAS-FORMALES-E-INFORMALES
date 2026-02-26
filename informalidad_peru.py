import streamlit as st
import pandas as pd
import joblib
import tensorflow as tf
import numpy as np
from PIL import Image

# 1. Configuración de Navegación y Estética
st.set_page_config(page_title="Tesis Informalidad UNMSM", layout="wide")

# Función para cargar modelos
@st.cache_resource
def load_assets():
    scaler = joblib.load('models/scaler.joblib')
    model_logit_py = joblib.load('models/modelo_logit.joblib')
    model_dl = tf.keras.models.load_model('models/model_dl.keras')
    return scaler, model_logit_py, model_dl

# --- LÓGICA DE PROCESAMIENTO PARA EL SIMULADOR ---
def procesar_inputs(edad, horas, genero, urbano, educ, idioma, cat_ocup, sector, tamanio):
    # Diccionario con las 27 columnas exactas de tu modelo
    input_data = {col: 0.0 for col in [
        'C318_T', 'EDAD', 'C207_2.0', 'NIVEL_EDUCATIVO_2_Primaria',
        'NIVEL_EDUCATIVO_3_Secundaria', 'NIVEL_EDUCATIVO_4_Superior_Tecnica',
        'NIVEL_EDUCATIVO_5_Superior_Universitaria', 'NIVEL_EDUCATIVO_6_Posgrado', 
        'IDIOMA_MATERNO_2_Lengua_Nativa', 'IDIOMA_MATERNO_3_Otros', 
        'C317_2.0', 'C317_3.0', 'C317_4.0', 'C317_5.0', 
        'MACRO_SECTOR_2_Mineria', 'MACRO_SECTOR_3_Manufactura',
        'MACRO_SECTOR_4_Electricidad_Agua', 'MACRO_SECTOR_5_Construccion',
        'MACRO_SECTOR_6_Comercio', 'MACRO_SECTOR_7_Servicios',
        'CATEGORIA_OCUPACIONAL_2_Independiente', 'CATEGORIA_OCUPACIONAL_3_Asalariado',
        'CATEGORIA_OCUPACIONAL_4_Trabajador_Familiar', 'CATEGORIA_OCUPACIONAL_5_Trabajador_Hogar',
        'CATEGORIA_OCUPACIONAL_6_Practicante', 'URBANO_2_Urbano_Intermedio', 'URBANO_3_Rural_Semi_Rural'
    ]}
    
    input_data['EDAD'] = edad
    input_data['C318_T'] = horas
    if genero == "Mujer": input_data['C207_2.0'] = 1
    if educ == "Primaria": input_data['NIVEL_EDUCATIVO_2_Primaria'] = 1
    elif educ == "Secundaria": input_data['NIVEL_EDUCATIVO_3_Secundaria'] = 1
    elif educ == "Superior Técnica": input_data['NIVEL_EDUCATIVO_4_Superior_Tecnica'] = 1
    elif educ == "Superior Universitaria": input_data['NIVEL_EDUCATIVO_5_Superior_Universitaria'] = 1
    elif educ == "Posgrado": input_data['NIVEL_EDUCATIVO_6_Posgrado'] = 1
    if idioma == "Lengua Nativa": input_data['IDIOMA_MATERNO_2_Lengua_Nativa'] = 1
    elif idioma == "Otros": input_data['IDIOMA_MATERNO_3_Otros'] = 1
    if tamanio == "Pequeña (11-50)": input_data['C317_2.0'] = 1
    elif tamanio == "Mediana (51-100)": input_data['C317_3.0'] = 1
    elif tamanio == "Grande (101+)": input_data['C317_4.0'] = 1
    elif tamanio == "Sector Público": input_data['C317_5.0'] = 1
    if sector == "Minería": input_data['MACRO_SECTOR_2_Mineria'] = 1
    elif sector == "Manufactura": input_data['MACRO_SECTOR_3_Manufactura'] = 1
    elif sector == "Electricidad/Agua": input_data['MACRO_SECTOR_4_Electricidad_Agua'] = 1
    elif sector == "Construcción": input_data['MACRO_SECTOR_5_Construccion'] = 1
    elif sector == "Comercio": input_data['MACRO_SECTOR_6_Comercio'] = 1
    elif sector == "Servicios": input_data['MACRO_SECTOR_7_Servicios'] = 1
    if cat_ocup == "Independiente": input_data['CATEGORIA_OCUPACIONAL_2_Independiente'] = 1
    elif cat_ocup == "Asalariado": input_data['CATEGORIA_OCUPACIONAL_3_Asalariado'] = 1
    elif cat_ocup == "Trabajador Familiar": input_data['CATEGORIA_OCUPACIONAL_4_Trabajador_Familiar'] = 1
    elif cat_ocup == "Trabajador Hogar": input_data['CATEGORIA_OCUPACIONAL_5_Trabajador_Hogar'] = 1
    elif cat_ocup == "Practicante": input_data['CATEGORIA_OCUPACIONAL_6_Practicante'] = 1
    if urbano == "Urbano Intermedio": input_data['URBANO_2_Urbano_Intermedio'] = 1
    elif urbano == "Rural/Semi-Rural": input_data['URBANO_3_Rural_Semi_Rural'] = 1
    
    return np.array([list(input_data.values())])

# --- PÁGINAS ---

def intro():
    st.title("Determinantes de la Informalidad Laboral en el Perú: Un enfoque de Aprendizaje Supervisado")
    st.subheader("Curso: Econometría III")
    st.divider()

    # Pestañas para organizar la información de la portada
    t1, t2, t3 = st.tabs(["👥 Miembros del Equipo", "📝 Resumen", "📋 Definición de Variables"])

    with t1:
        st.markdown("""
        ### Integrantes:
        * Luis Mauricio Aguirre Stornaiuolo
        * Tilsa Morgana Tejeda Becerra
        * Gary Magno Alca Chipana
        * Edwin Joel Quispe Mamani
        
        **Institución:** Universidad Nacional Mayor de San Marcos (FCE)
        """)

    with t2:
        st.markdown("""
        Este proyecto de investigación analiza la probabilidad de informalidad laboral en el Perú utilizando microdatos de la **EPEN 2024**.
        El objetivo central es comparar la capacidad explicativa de los modelos **Logit** tradicionales frente a la precisión predictiva de modelos de **Deep Learning**.
        
        A través de este aplicativo, se presentan los resultados de inferencia, las métricas de rendimiento y un simulador interactivo para evaluar perfiles de trabajadores en tiempo real.
        """)

    with t3:
        st.markdown("### Diccionario de Variables Seleccionadas")
        st.markdown("""
        | Variable | Descripción | Fuente |
        | :--- | :--- | :--- |
        | **Informal** | Condición de informalidad del trabajador (Variable dependiente). | EPEN |
        | **Edad** | Edad cronológica del trabajador y su término cuadrático. | C208 |
        | **Género** | Identificación de género (Dicotómica: Hombre/Mujer). | C207 |
        | **Educación** | Nivel educativo alcanzado (Desde sin nivel hasta posgrado). | C366 |
        | **Sector** | Macro sectores económicos (Agricultura, Minería, Servicios, etc.). | C309 |
        | **Empresa** | Tamaño de la unidad económica según número de trabajadores. | C317 |
        | **Urbano** | Ubicación geográfica según densidad poblacional. | Estrato |
        """)

def eda_page():
    st.title("📊 Análisis Exploratorio de Datos (EDA)")
    st.write("Visualizaciones clave de la muestra de 91,725 observaciones.")
    
    t1, t2, t3 = st.tabs(["Distribución de Informalidad", "Relación Educación/Edad", "Análisis Sectorial"])
    
    with t1:
        st.subheader("Prevalencia de la Informalidad")
        try:
            st.image('data/eda_target.png', caption="Distribución de la variable dependiente (Formal vs Informal)")
        except:
            st.info("💡 Sugerencia: Carga un gráfico de barras mostrando la proporción de informales en la muestra.")
            
    with t2:
        st.subheader("Informalidad por Nivel Educativo")
        try:
            st.image('data/eda_educacion.png', caption="Tasa de informalidad según grado académico")
        except:
            st.info("💡 Sugerencia: Un gráfico de barras apiladas (stacked bar chart) de Educación vs Informalidad.")
            
    with t3:
        st.subheader("Mapa de Calor de Correlaciones")
        try:
            st.image('data/eda_heatmap.png', caption="Correlación entre variables numéricas")
        except:
            st.info("💡 Sugerencia: Un Heatmap para mostrar la relación entre Edad, Horas y la Informalidad.")

def inferencia_stata():
    st.title("📈 3. Resultados Econométricos (Stata)")
    
    # Tres pestañas bien definidas
    tab1, tab2, tab3 = st.tabs(["Odds Ratios (OR)", "Efectos Marginales", "Gráfico de Coeficientes"])
    
    with tab1:
        st.write("### Modelo Logit: Interpretación por Odds Ratios")
        try:
            # Cargamos el archivo que guardaste como CSV desde el XLS de Odds Ratios
            df_or = pd.read_csv('data/resultados_or.csv')
            df_or.columns = ['Variables', 'Odds Ratio'] # Renombramos para que sea claro
            
            # Limpieza profesional:
            # 1. Cortar después de las variables (antes de 'Observations')
            if 'Observations' in df_or['Variables'].values:
                df_or = df_or.iloc[:df_or[df_or['Variables'] == 'Observations'].index[0]]
            
            # 2. Eliminar filas de errores estándar (las que tienen paréntesis)
            df_or_clean = df_or[~df_or['Variables'].isna()]
            df_or_clean = df_or_clean[~df_or_clean['Odds Ratio'].astype(str).str.contains(r'\(', na=False)]
            
            st.dataframe(df_or_clean, use_container_width=True, hide_index=True)
            st.info("💡 Un OR > 1 indica que la variable aumenta la probabilidad de informalidad; un OR < 1 indica que la reduce.")
            
        except:
            st.warning("Carga 'resultados_or.csv' en la carpeta data.")

    with tab2:
        st.write("### Efectos Marginales (Promedio)")
        try:
            # Cargamos el archivo de efectos marginales
            df_m = pd.read_csv('data/efectos_marginales.csv')
            df_m.columns = ['Variables', 'Efecto Marginal (dy/dx)']
            
            # Aplicamos la misma limpieza
            if 'Observations' in df_m['Variables'].values:
                df_m = df_m.iloc[:df_m[df_m['Variables'] == 'Observations'].index[0]]
            
            df_m_clean = df_m[~df_m['Variables'].isna()]
            df_m_clean = df_m_clean[~df_m_clean['Efecto Marginal (dy/dx)'].astype(str).str.contains(r'\(', na=False)]
            
            st.dataframe(df_m_clean, use_container_width=True, hide_index=True)
            st.success("✅ Los efectos marginales permiten cuantificar el impacto directo en puntos porcentuales.")
        except:
            st.warning("Carga 'efectos_marginales.csv' en la carpeta data.")

    with tab3:
        st.write("### Visualización de Coeficientes")
        try:
            st.image('data/grafico_coeficientes.png', use_container_width=True)
        except:
            st.error("No se encontró el archivo 'grafico_coeficientes.png'.")


def modelo_logit_py():
    st.title("🐍 4. Modelo Logit (Python - Sklearn)")
    scaler, model_logit, _ = load_assets()
    
    t1, t2 = st.tabs(["Métricas y Evaluación", "Simulador Interactivo"])
    
    with t1:
        st.write("### Reporte de Clasificación - Sklearn")
        # Mostramos el reporte en formato código para mantener la estructura
        st.code("""
               precision    recall  f1-score   support

           0       0.75      0.79      0.77      6155
           1       0.90      0.87      0.88     12531

    accuracy                           0.84     18686
   macro avg       0.82      0.83      0.83     18686
weighted avg       0.85      0.84      0.85     18686
        """)
        
        c1, c2 = st.columns(2)
        with c1:
            try:
                st.image('data/matriz_logit.png', caption="Matriz de Confusión Logit")
            except:
                st.warning("Archivo 'matriz_logit.png' no encontrado.")
        with c2:
            try:
                st.image('data/roc_logit.png', caption="Curva ROC Logit")
            except:
                st.warning("Archivo 'roc_logit.png' no encontrado.")
            
    with t2:
        render_simulador(scaler, model_logit, "LOGIT")


def modelo_dl_page():
    st.title("🧠 5. Modelo Deep Learning")
    scaler, _, model_dl = load_assets()
    
    t1, t2 = st.tabs(["Métricas y Evaluación", "Simulador Interactivo"])
    
    with t1:
        st.write("### Reporte de Clasificación - Keras/TensorFlow")
        st.code("""
                  precision    recall  f1-score   support

      Formal (0)       0.81      0.76      0.79      6155
    Informal (1)       0.89      0.91      0.90     12531

        accuracy                           0.86     18686
       macro avg       0.85      0.84      0.84     18686
    weighted avg       0.86      0.86      0.86     18686
        """)
        
        c1, c2 = st.columns(2)
        with c1:
            try:
                st.image('data/matriz_dl.png', caption="Matriz de Confusión DL")
            except:
                st.warning("Archivo 'matriz_dl.png' no encontrado.")
        with c2:
            st.success("**Análisis:** El modelo de Deep Learning logra un F1-Score superior para la clase informal (0.90), reduciendo significativamente los errores de tipo II.")
            
    with t2:
        render_simulador(scaler, model_dl, "DL")

def render_simulador(scaler, model, m_type):
    st.info(f"Simulador basado en el modelo {m_type}")
    c1, c2, c3 = st.columns(3)
    with c1:
        edad = st.number_input(f"Edad", 14, 95, 25, key=f"ed_{m_type}")
        horas = st.number_input(f"Horas semanales", 1, 112, 40, key=f"hr_{m_type}")
        genero = st.selectbox(f"Género", ["Hombre", "Mujer"], key=f"gen_{m_type}")
        urb = st.selectbox(f"Zona", ["Metropolitana", "Urbano Intermedio", "Rural/Semi-Rural"], key=f"urb_{m_type}")
    with c2:
        educ = st.selectbox(f"Educación", ["Sin Nivel/Inicial", "Primaria", "Secundaria", "Superior Técnica", "Superior Universitaria", "Posgrado"], key=f"edc_{m_type}")
        idi = st.selectbox(f"Idioma", ["Castellano", "Lengua Nativa", "Otros"], key=f"id_{m_type}")
        cat = st.selectbox(f"Categoría Ocupacional", ["Empleador", "Independiente", "Asalariado", "Trabajador Familiar", "Trabajador Hogar", "Practicante"], key=f"cat_{m_type}")
    with c3:
        sec = st.selectbox(f"Sector", ["Agri/Pesca", "Minería", "Manufactura", "Electricidad/Agua", "Construcción", "Comercio", "Servicios"], key=f"sec_{m_type}")
        tam = st.selectbox(f"Tamaño Empresa", ["Micro (1-10)", "Pequeña (11-50)", "Mediana (51-100)", "Grande (101+)", "Sector Público"], key=f"tam_{m_type}")

    if st.button(f"Predecir Informalidad ({m_type})"):
        # 1. Obtenemos el vector base de 27 columnas (con EDAD y C318_T sin escalar)
        features = procesar_inputs(edad, horas, genero, urb, educ, idi, cat, sec, tam)
        
        # 2. Preparamos las 3 variables que el scaler espera: ["EDAD", "EDAD^2", "C318_T"]
        # Respetamos el orden exacto del fit_transform de tu Python
        edad2 = edad**2
        inputs_para_escalar = np.array([[edad, edad2, horas]])
        
        # Escalamiento
        scaled_values = scaler.transform(inputs_para_escalar)
        
        # 3. Insertamos los valores escalados en el vector de 27 columnas
        # Según tu Index: 'C318_T' es posición 0, 'EDAD' es posición 1
        features_finales = features.copy()
        features_finales[0, 1] = scaled_values[0, 0] # EDAD escalada
        features_finales[0, 0] = scaled_values[0, 2] # C318_T escalada
        # Nota: EDAD^2 (scaled_values[0, 1]) se ignora porque la dropeaste de data_final
        
        # 4. Predicción según el modelo
        if m_type == "LOGIT":
            prob = model.predict_proba(features_finales)[0][1]
        else:
            # Para Keras/DL
            prob = model.predict(features_finales, verbose=0)[0][0]
            
        st.metric("Probabilidad de Informalidad", f"{prob*100:.2f}%")
        
        if prob > 0.5:
            st.error("Predicción: **INFORMAL**")
        else:
            st.success("Predicción: **FORMAL**")



def despedida():
    st.title("🏁 Conclusiones y Cierre")
    st.divider()
    
    # Apartado de Conclusiones con énfasis en los resultados obtenidos
    st.markdown("""
    ### Conclusiones del Estudio
    * **Desempeño Comparativo:** El modelo de **Deep Learning** superó al Logit tradicional en capacidad predictiva (86% vs 84% de accuracy). La arquitectura de la red neuronal permitió capturar interacciones complejas entre la edad, educación y sector.
    * **Focalización del Target:** Se logró un **Recall de 0.91** en el modelo DL para la clase informal, lo que sugiere que estas herramientas son altamente robustas para el diseño de políticas de formalización dirigidas.
    * **Determinantes Estructurales:** Tanto en Stata como en Python, se confirma que la educación superior y el tamaño de la unidad económica son los factores con mayor impacto en la reducción de la probabilidad de informalidad.
    * **Innovación Metodológica:** La integración de microdatos de la **EPEN 2024** con técnicas de aprendizaje supervisado representa un avance significativo para la inferencia y predicción en la economía laboral peruana.
    """)
    
    st.divider()

    # Apartado de Despedida y Enlaces
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Foto de perfil de GitHub (usando tu link directo)
        github_url = "https://github.com/MrPinkiePie"
        st.image(f"{github_url}.png", width=220)
        
    with col2:
        st.markdown(f"""
        ### ¡Gracias por su atención!
        Este proyecto fue desarrollado para el curso de **Econometría III** por el grupo conformado por:
        * Luis Mauricio Aguirre Stornaiuolo
        * Tilsa Morgana Tejeda Becerra
        * Gary Magno Alca Chipana
        * Edwin Joel Quispe Mamani
        
        Sientanse libres de hacer cualquier pregunta o resolver alguna duda adicional.
        """)
        
        # Botón simplificado
        st.link_button("🚀 Visitar GitHub", github_url, use_container_width=True)
        
        st.caption("Repositorio con microdatos, documentación y modelos exportados.")

    st.balloons()
# --- NAVEGACIÓN ---
pg = st.navigation([
    st.Page(intro, title="1. Portada", icon="🏠"),
    st.Page(eda_page, title="2. Análisis Exploratorio", icon="📊"),
    st.Page(inferencia_stata, title="3. Resultados Stata", icon="📈"),
    st.Page(modelo_logit_py, title="4. Logit Python", icon="🐍"),
    st.Page(modelo_dl_page, title="5. Deep Learning", icon="🧠"),
    st.Page(despedida, title="6. Finalizar", icon="✨") # Nueva página de cierre
])

pg.run()















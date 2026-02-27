# 🇵🇪 Análisis de la Informalidad Laboral en el Perú (EPEN 2024)

Este repositorio contiene el proyecto final para el curso de **Econometría III** de la **Universidad Nacional Mayor de San Marcos**. El estudio utiliza microdatos de la **Encuesta Permanente de Empleo Nacional (EPEN) 2024** para predecir la probabilidad de que un trabajador pertenezca al sector informal, comparando modelos econométricos tradicionales con algoritmos de **Aprendizaje Supervisado**.

## 👥 Equipo de Investigación (9no Ciclo - UNMSM)
* **Luis Mauricio Aguirre Stornaiuolo**
* **Tilsa Morgana Tejeda Becerra**
* **Gary Magno Alca Chipana**
* **Edwin Joel Quispe Mamani**

---

## 🎯 Objetivo del Proyecto
El objetivo central es identificar los determinantes de la informalidad laboral mediante un enfoque híbrido:
1. **Inferencia Econométrica:** Implementación de modelos **Logit** en Stata y Python para el análisis de coeficientes (Odds Ratios) y efectos marginales.
2. **Capacidad Predictiva:** Uso de **Deep Learning** (Redes Neuronales Densas) para capturar interacciones no lineales y mejorar la precisión en la identificación de trabajadores en riesgo.



---

## 🛠️ Stack Tecnológico
* **Econometría:** Stata (Inferencia y Coefplot).
* **Lenguaje:** Python 3.x.
* **Librerías de ML:** Scikit-Learn (Logit, Scaler), TensorFlow/Keras (Redes Neuronales).
* **Visualización:** Seaborn, Matplotlib, PIL.
* **Despliegue:** Streamlit Cloud.

---

## 🚀 Acceso al Proyecto

### 1. Cuaderno de Investigación (Google Colab)
Puedes revisar el flujo completo de limpieza de datos, ingeniería de variables (incluyendo el tratamiento de la **Edad al cuadrado**) y entrenamiento de modelos aquí:
👉 [**Abrir Notebook en Google Colab**](https://colab.research.google.com/drive/1sPSRcPvBlQkPgZHhlLXG9ihv5Dx2hfIj?usp=sharing)

### 2. Aplicativo Interactivo (Streamlit)
Hemos desarrollado un simulador en tiempo real que permite calcular la probabilidad de informalidad según el perfil del trabajador:
👉 **[Enlace a la Web App aquí]** *(Pega tu URL de Streamlit aquí)*

---

## 📊 Principales Hallazgos
* **Educación y Capital Humano:** El nivel educativo superior reduce significativamente la probabilidad de informalidad.
* **Desempeño del Modelo:** La Red Neuronal alcanzó un **Recall de 0.91** para la clase informal, superando al modelo Logit tradicional en la detección de casos positivos.
* **Impacto del Tamaño de Empresa:** Las microempresas presentan los mayores niveles de riesgo de informalidad laboral.

---

## 📁 Estructura del Repositorio
* `informalidad_peru.py`: Código principal del aplicativo Streamlit.
* `requirements.txt`: Dependencias necesarias para el entorno de producción.
* `data/`: Resultados exportados de Stata (CSVs) y gráficos de análisis exploratorio (EDA).
* `models/`: Modelos entrenados (`.joblib` y `.keras`) y el escalador de variables numéricas.

---
*Este proyecto es parte del portafolio académico de Luis Mauricio Aguirre Stornaiuolo enfocado en la aplicación de Data Science en la Economía y Finanzas.*

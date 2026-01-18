# Sistema de Clasificación y Segmentación de Fracturas en Radiografías

Este repositorio contiene el código fuente desarrollado para un modelo de
clasificación y segmentación automática de fracturas óseas en imágenes
radiográficas del brazo, como parte de un trabajo académico aplicado.

El sistema integra dos etapas principales:
1. Clasificación binaria de imágenes (radiografía / no radiografía).
2. Segmentación semántica de la región de fractura mediante U-Net.

---

## 📌 Tecnologías utilizadas

- Python 3.11+
- TensorFlow / Keras
- PyTorch (opcional según módulo)
- OpenCV
- NumPy
- Matplotlib
- Visual Studio Code
- Roboflow
- Kaggle

---

## 📂 Estructura del proyecto

- `segmentation/`: scripts para entrenamiento y predicción del modelo U-Net.
- `data/`: estructura del conjunto de datos (entrenamiento, validación y prueba).
- `requirements.txt`: dependencias necesarias para la ejecución del proyecto.

---

## ⚙️ Instalación del entorno de interfaz web 

Se recomienda el uso de un entorno virtual.

```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
!pip install streamlit

## Levantar la interfaz web
streamlit run completo.py



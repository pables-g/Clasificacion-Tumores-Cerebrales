# Clasificación de Tumores Cerebrales mediante Redes Neuronales Convolucionales (CNN)

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)](https://www.tensorflow.org/)

Este repositorio contiene la implementación en Python de una red neuronal convolucional (CNN) para la clasificación de tumores cerebrales a partir de imágenes de resonancia magnética (IRM). Desarrollado como Trabajo de Fin de Grado (TFG) en Matemáticas.

## 🛠️ Características principales

- Clasificación en 4 categorías: Glioma, Meningioma, Tumor pituitario y Sin tumor
- Arquitectura CNN (personalizable) con:
  - Capas convolucionales y de pooling
  - Batch Normalization
  - Dropout para regularización
- Evaluación con métricas:
  - Accuracy, Precision, Recall, F1-score
  - Matriz de confusión
  - Curva ROC (micro-averaged OvR) para calcular AUC AUC

## 📦 Requisitos

- Python 3.8+
- Librerias utilizadas:
   - Kagglehub
   - Os
   - Cv2
   - Numpy
   - Tqdm
   - Sklearn
   - Time
   - Matplotlib
   - Tensorflow

## 🏗️ **Estructura del proyecto**

brain-tumor-classification/
├── data_loader.py    # Carga y preprocesamiento de datos
├── cnn.py            # Implementación de la CNN
├── main.py           # Script principal
├── brain tumor.keras # Modelo pre-entrenado
├── requirements.txt  # Dependencias
└── README.md         # Este archivo

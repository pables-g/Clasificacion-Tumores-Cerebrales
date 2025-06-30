# Clasificación de Tumores Cerebrales mediante Redes Neuronales Convolucionales (CNN)

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

Este repositorio contiene la implementación en Python de una red neuronal convolucional (CNN) para la clasificación de tumores cerebrales a partir de imágenes de resonancia magnética (IRM). Desarrollado como Trabajo de Fin de Grado (TFG) en Matemáticas.

## � Características principales

- Clasificación en 4 categorías: Glioma, Meningioma, Tumor pituitario y Sin tumor
- Arquitectura CNN personalizable con:
  - Capas convolucionales y de pooling
  - Batch Normalization
  - Dropout para regularización
- Evaluación con métricas completas:
  - Accuracy, Precision, Recall
  - Matriz de confusión
  - Curva ROC y AUC

## 📦 Requisitos

- Python 3.8+
- Dependencias:
  ```bash
  pip install tensorflow opencv-python numpy scikit-learn matplotlib kagglehub tqdm

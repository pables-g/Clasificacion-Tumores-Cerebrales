# Clasificación de Tumores Cerebrales con Redes Neuronales Convolucionales (CNN)

Este repositorio contiene el código fuente desarrollado en Python para el Trabajo de Fin de Grado en Matemáticas: **"Clasificación de tumores mediante el uso de redes neuronales"**, en la Universidad Complutense de Madrid durante el curso académico 2024–2025.

## 📄 Descripción del Proyecto

El proyecto tiene como objetivo implementar y entrenar una red neuronal convolucional (CNN) para clasificar imágenes de resonancias magnéticas cerebrales en cuatro categorías:

- Glioma
- Meningioma
- Tumor pituitario
- Sin tumor

El modelo se entrena utilizando conjuntos de datos públicos ([base de datos disponible en Kaggle]( https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri)) y se evalúa mediante distintas métricas de clasificación. Se emplean técnicas de Deep Learning y se aplica regularización para evitar sobreajuste.

## 🧠 Arquitectura del modelo

- Red CNN con múltiples capas de convolución y pooling
- Capas Fully Connected al final
- Función de activación `ReLU`
- Capa de salida `Softmax` para clasificación multiclase
- Optimizador: `Adam`
- Función de pérdida: `categorical_crossentropy`

## 📁 Estructura del repositorio

```
.
├── README.md            # Este archivo
└── CODIGO
    ├── data_loader.py   # Funciones para carga, preprocesamiento y división del dataset
    ├── cnn.py           # Definición de la red neuronal CNN
    └── main.py          # Scrip principal en el que se usan las clases Data y CNN
```

## 🛠️ Requisitos

- Python
- TensorFlow
- NumPy
- Pandas
- scikit-learn
- OpenCV (`cv2`)
- tqdm
- matplotlib

## ⚙️ Cómo ejecutar

1. Coloca las imágenes en las carpetas adecuadas (`Training` y `Testing`) siguiendo la estructura esperada por `data_loader.py`.
2. Ejecuta el flujo (básico) con:

```bash
python main.py
```

## 📈 Resultados

| Clase       | Precision | Recall    | F1-score  | Support   |
|-------------|-----------|-----------|-----------|-----------|
| Glioma      | 0.87      | 0.91      | 0.89      | 93        |
| No tumor    | 0.92      | 0.92      | 0.92      | 51        |
| Meningioma  | 0.92      | 0.88      | 0.90      | 96        |
| Pituitario  | 0.99      | 0.99      | 0.99      | 87        |

| Métricas    | Valor     |
|-------------|-----------|
| Func. coste | 0.36      |
| Accuracy    | 0.92      |
| AUC         | 0.99      |

## 👨‍🎓 Autor

**Pablo García Hernández**  
Grado en Matemáticas  
Universidad Complutense de Madrid

# Clasificación de Tumores Cerebrales con Redes Neuronales Convolucionales (CNN)

Este repositorio contiene el código fuente desarrollado en Python para el Trabajo de Fin de Grado en Matemáticas: **"Clasificación de tumores mediante el uso de redes neuronales"**, en la Universidad Complutense de Madrid durante el curso académico 2024–2025.

## 📄 Descripción del Proyecto

El proyecto tiene como objetivo implementar y entrenar una red neuronal convolucional (CNN) para clasificar imágenes de resonancias magnéticas cerebrales en cuatro categorías:

- Glioma
- Meningioma
- Tumor pituitario
- Sin tumor

El modelo se entrena utilizando conjuntos de datos públicos ([base de datos disponible en Kaggle]( https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri)) y se evalúa mediante distintas métricas de clasificación. Se emplean técnicas modernas de Deep Learning y se aplica regularización para evitar sobreajuste.

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

- Python ≥ 3.7
- TensorFlow ≥ 2.0
- NumPy
- Pandas
- scikit-learn
- OpenCV (`cv2`)
- tqdm
- matplotlib

## ⚙️ Cómo ejecutar

1. Coloca las imágenes en las carpetas adecuadas (`train` y `test`) siguiendo la estructura esperada por `data_loader.py`.
2. Ejecuta el flujo (básico) con:

```bash
python main.py
```

## 📈 Resultados

El modelo alcanzó una **precisión superior al 90%** en el conjunto de prueba. Los resultados demuestran la capacidad de las CNNs para detectar patrones complejos en imágenes médicas, mostrando su potencial como herramienta de apoyo al diagnóstico clínico.

## 📚 Referencias

Este trabajo se basa en el TFG disponible en PDF: [TFG_PABLO_GARCIA_HERNANDEZ.pdf](./TFG_PABLO_GARCIA_HERNANDEZ.pdf)

## 👨‍🎓 Autor

**Pablo García Hernández**  
Grado en Matemáticas  
Universidad Complutense de Madrid

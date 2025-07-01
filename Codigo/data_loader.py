import kagglehub
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import tensorflow as tf

class Data:
    def __init__(self, url : str , image_size : int, labels : list) -> None:
        self.url = url # "sartajbhuvaji/brain-tumor-classification-mri"
        self.image_size = image_size # 256
        self.labels = labels # ['glioma_tumor','no_tumor','meningioma_tumor','pituitary_tumor']

    def load_data(self, path : str ):
        return self.training_sets(path)

    def download(self) -> str:
        path = kagglehub.dataset_download(self.url)
        print("Path to dataset files:", path)
        return path
    
    def training_sets(self, path : str):
        X_train = []
        y_train = []
        for i in self.labels: # Cargar el training
            folderPath = os.path.join(path + '/Training',i)
            for j in tqdm(os.listdir(folderPath)):
                img = cv2.imread(os.path.join(folderPath,j))
                img = cv2.resize(img,(self.image_size, self.image_size))
                X_train.append(img)
                y_train.append(i)
        for i in self.labels: # Cargar el testing
            folderPath = os.path.join(path + '/Testing',i)
            for j in tqdm(os.listdir(folderPath)):
                img = cv2.imread(os.path.join(folderPath,j))
                img = cv2.resize(img,(self.image_size,self.image_size))
                X_train.append(img)
                y_train.append(i)
        X_train, y_train = np.array(X_train), np.array(y_train) # Convertir a numpy arrays
        X_train,y_train = shuffle(X_train,y_train,random_state=101) # Mezclar los datos
        X_train,X_test,y_train,y_test = train_test_split(X_train,y_train,test_size=0.1,random_state=101) # Dividir los datos

        #Para utilizar la funcion plot, comentar con """ desde aqui
        y_train_new = []
        for i in y_train:
            y_train_new.append(self.labels.index(i))
            y_train = y_train_new
            y_train = tf.keras.utils.to_categorical(y_train)
        y_test_new = []
        for i in y_test:
            y_test_new.append(self.labels.index(i))
            y_test = y_test_new
            y_test = tf.keras.utils.to_categorical(y_test)
        X_train, X_test = X_train / 255.0, X_test / 255.0
        #Para utilizar la funcion plot, comentar con """ hasta aquí

        return X_train,X_test,y_train,y_test
    
    def plot(self, X: list, y: list) -> None:
        label_counts = {label: np.sum(y == label) for label in self.labels}
        plt.figure(figsize=(8, 6))
        colors = ["C0", "C1", "C2", "C3"]
        plt.subplot(2, 1, 1)
        plt.bar(label_counts.keys(), label_counts.values(), color=colors)
        plt.ylabel('Count')
        plt.title('Distribution of Labels')
        k = 0
        for i in self.labels:
            j = 0
            while True:
                if y[j] == i:
                    plt.subplot(2, 4, k + 5) 
                    plt.imshow(X[j])
                    plt.axis('off')
                    k += 1
                    break
                j += 1
                
        plt.tight_layout()
        plt.show()

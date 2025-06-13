import tensorflow as tf
from time import perf_counter
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report,confusion_matrix, ConfusionMatrixDisplay

class CNN:
    def __init__(self, input_shape, feature_map, num_classes):
        self.input_shape = input_shape
        self.feature_map = feature_map
        self.num_classes = num_classes
        self.model = tf.keras.models.Sequential()

    def create_network(self) -> None:
        """
        self.model.add(tf.keras.layers.Input(shape=(256, 256, 3)))

        self.model.add(tf.keras.layers.Conv2D(32, kernel_size=(5,5), activation='relu', padding='same'))
        self.model.add(tf.keras.layers.MaxPooling2D())
        self.model.add(tf.keras.layers.Dropout(0.15))

        self.model.add(tf.keras.layers.Conv2D(32, kernel_size=(5,5), activation='relu', padding='same'))
        self.model.add(tf.keras.layers.MaxPooling2D())
        self.model.add(tf.keras.layers.Dropout(0.15))

        self.model.add(tf.keras.layers.Conv2D(32, kernel_size=(3,3), activation='relu', padding='same'))
        self.model.add(tf.keras.layers.MaxPooling2D())
        self.model.add(tf.keras.layers.Dropout(0.15))

        self.model.add(tf.keras.layers.Conv2D(64, kernel_size=(2,2), activation='relu', padding='same'))
        self.model.add(tf.keras.layers.MaxPooling2D())
        self.model.add(tf.keras.layers.Dropout(0.15))
            
        self.model.add(tf.keras.layers.Flatten())
        self.model.add(tf.keras.layers.Dense(128, activation='relu'))
        self.model.add(tf.keras.layers.Dropout(0.45))
        self.model.add(tf.keras.layers.Dense(4, activation='softmax'))
        """
        self.model.add(tf.keras.layers.Input(shape = (self.input_shape, self.input_shape, 3)))
        self.model.add(tf.keras.layers.Conv2D(32,kernel_size = (5,5), activation='relu', padding = 'same'))
        self.model.add(tf.keras.layers.BatchNormalization())
        self.model.add(tf.keras.layers.MaxPooling2D())
        self.model.add(tf.keras.layers.Dropout(rate = 0.15))
        i = 32
        for _ in range(self.feature_map):
            self.model.add(tf.keras.layers.Conv2D(i,kernel_size = (3,3), activation='relu', padding = 'same'))
            self.model.add(tf.keras.layers.MaxPooling2D())
            self.model.add(tf.keras.layers.Dropout(rate = 0.15))
            i *= 2
        self.model.add(tf.keras.layers.Flatten())
        self.model.add(tf.keras.layers.Dense(128,activation='relu'))
        self.model.add(tf.keras.layers.Dense(128,activation='relu'))
        self.model.add(tf.keras.layers.Dropout(rate = 0.3))
        self.model.add(tf.keras.layers.Dense(self.num_classes,activation='softmax'))

    def compile_network(self) -> None:
        self.model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy','precision','auc'])

    def train_network(self, X_train : list, y_train : list, epochs : int, batch_size : int) -> None:
        start_time = perf_counter()
        history = self.model.fit(X_train, y_train, epochs = epochs, validation_split = 0.1, batch_size = batch_size)
        end_time = perf_counter()
        print("Total runtime:", round(end_time - start_time,4), "seconds")
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        epochs = range(len(acc))
        fig = plt.figure(figsize=(14,7))
        plt.plot(epochs,acc,'r',label="Training Accuracy")
        plt.plot(epochs,val_acc,'b',label="Validation Accuracy")
        plt.legend(loc='upper left')
        plt.show()
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs = range(len(loss))
        fig = plt.figure(figsize=(14,7))
        plt.plot(epochs,loss,'r',label="Training loss")
        plt.plot(epochs,val_loss,'b',label="Validation loss")
        plt.legend(loc='upper left')
        plt.show()

    def load_network(self, modelo : str) -> None:
        self.create_network()
        self.model = tf.keras.models.load_model(modelo)

    def architecture_network(self) -> None:
        self.model.summary()

    def confusion_matrix(self, X_train: list, y_train: list) -> None:
        y_true = np.argmax(y_train, axis=1)
        y_pred = np.argmax(self.model.predict(X_train), axis=1)
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(20, 25))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['glioma','no_tumor', 'meningioma','pituary_tumor'])
        disp.plot(cmap=plt.cm.Blues, values_format='d')
    
        plt.xlabel('Valores_predichos', fontsize=11)
        plt.ylabel('Valores_reales', fontsize=11)
        plt.tight_layout() 
        plt.show()

    def test_network(self, X_test : list, y_test : list) -> None:
        scores = self.model.evaluate(X_test, y_test)
        print(f"Loss: {round(scores[0],4)}, Accuracy: {round(scores[1],4)}, AUC: {round(scores[2],4)}")
        y_true_test = np.argmax(y_test, axis=1)
        y_pred_test = np.argmax(self.model.predict(X_test), axis=1) 
        print(classification_report(y_true_test,y_pred_test))

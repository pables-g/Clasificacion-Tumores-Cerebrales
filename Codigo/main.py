from data_loader import Data
from cnn import CNN

if __name__ == "__main__":
    dl = Data("sartajbhuvaji/brain-tumor-classification-mri", 256, ['glioma_tumor','no_tumor','meningioma_tumor','pituitary_tumor'])
    cnn = CNN(256, 3, 4)
    """
    path = dl.download()
    X_train, X_test, y_train, y_test = dl.training_sets(path)
    X_train, X_test, y_train, y_test = dl.load_data('/Users/pablogarcia/.cache/kagglehub/datasets/sartajbhuvaji/brain-tumor-classification-mri/versions/2')
    #dl.plot(X_train, y_train)

    cnn.create_network()
    cnn.compile_network()
    cnn.architecture_network()
    cnn.train_network(X_train, y_train, epochs = 10, batch_size = 16)
    
    cnn.model.save('braintumor3.keras')
    cnn.test_network(X_test,y_test)
    """
    X_train, X_test, y_train, y_test = dl.load_data('/Users/pablogarcia/.cache/kagglehub/datasets/sartajbhuvaji/brain-tumor-classification-mri/versions/2')
    cnn.load_network('braintumor3.keras')
    cnn.architecture_network()
    cnn.confusion_matrix(X_train,y_train)
    cnn.calculate_multiclass_auc(X_test, y_test)
    cnn.test_network(X_test,y_test)

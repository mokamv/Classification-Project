import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelBinarizer
from sklearn import metrics




with open('Task 1/iris.data', 'r') as file: 
    # Read the contents of the file 
    data1 = file.readlines()


data = [line.strip().split(',') for line in data1 if line.strip() != '']


x_data = [[float(value) for value in row[:-1]] for row in data]


x_data = np.array(x_data, dtype=np.float32)


#Appending ones to training set and test, to represent bias
x_data = np.array([row + [1] for row in x_data], dtype=np.float32)


#Extracting the y-labels fron the dat set, these are the 
y_labels = [row[-1] for row in data]
    


#Preprossessing of data: We give each flower one-hot encoded label
lb = LabelBinarizer()
lb.fit(y_labels)
labels = lb.transform(y_labels)



#Splitting of data into a test set and a training set
X_train = np.concatenate([x_data[0:30], x_data[50:80], x_data[100:130]])
T_train = np.concatenate([labels[0:30], labels[50:80], labels[100:130]])

X_test = np.concatenate([x_data[30:50], x_data[80:100], x_data[130:150]])
T_test = np.concatenate([labels[30:50], labels[80:100], labels[130:150]])



def sigmoid(x):
    return 1 / (1 + np.exp(-x))


#Computing the MSE gradient

def compute_gradients(X, G, T):
    delta = (G - T) * G * (1 - G)     # shape: (N, C)
    grad_W = delta.T @ X             # (C, N) @ (N, D+1) = (C, D+1)
    return grad_W

def compute_mse(G, T):
    return 0.5 * np.mean(np.sum((G - T)**2, axis=1))


def train_linear_classifier(X, T, num_classes, num_epochs=1000, alpha=0.005):
    N, D_plus_1 = X.shape
    W = np.random.randn(num_classes, D_plus_1) * 0.01  # weight
    loss_history = []
    patience = 10
    min_delta = 0.0005  # Minimum change in loss to be considered an improvement
    best_loss = float('inf')
    wait = 0

    for epoch in range(num_epochs):
        Z = X @ W.T
        G = sigmoid(Z)
        grad_W = compute_gradients(X, G, T)
        W -=alpha* grad_W
        loss = compute_mse(G,T)
        loss_history.append(loss)
        targets = np.argmax(T, axis=1)
        predictions = np.argmax(G, axis=1)



        if best_loss - loss > min_delta:
            best_loss = loss
            wait = 0
        else:
            wait += 1

        print(f"Epoch {epoch} - Val Loss: {loss:.4f} - Best: {best_loss:.4f} - Wait: {wait}")

        if wait >= patience:
            print("Stopping early due to no improvement.")
            break
    
        
    return W, loss_history, targets, predictions

W_trained, loss_history, tr_targets, tr_predictions = train_linear_classifier(X_train, T_train, num_classes=3, num_epochs=500, alpha=0.005)

W_test, test_loss_history, test_targets, test_predictions = train_linear_classifier(X_test, T_test, num_classes=3, num_epochs=500, alpha=0.005)

#Confusion matrices for training set

train_confusion_matrix = metrics.confusion_matrix(tr_targets, tr_predictions)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = train_confusion_matrix, display_labels = [0, 1, 2])
cm_display.plot()
plt.show()

#Error rate for training set

accuracy = np.mean(tr_targets == tr_predictions)
print({accuracy:.2f})


#Confusion matrices and errror rate for training set
test_confusion_matrix = metrics.confusion_matrix(test_targets, test_predictions)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = test_confusion_matrix, display_labels = [0, 1, 2])
cm_display.plot()
plt.show()

#Error rate for test set
accuracy = np.mean(test_targets == test_predictions)
print(accuracy:.2f)



X_train2 = np.concatenate([x_data[0:20], x_data[50:70], x_data[100:120]])
T_train2 = np.concatenate([labels[0:20], labels[50:70], labels[100:120]])

X_test2 = np.concatenate([x_data[20:50], x_data[70:100], x_data[120:150]])
T_test2 = np.concatenate([labels[20:50], labels[70:100], labels[120:150]])

W_trained2, loss_history, tr_targets2, tr_predictions2 = train_linear_classifier(X_train2, T_train2, num_classes=3, num_epochs=500, alpha=0.005)

W_test2, test_loss_history, test_targets2, test_predictions2 = train_linear_classifier(X_test2, T_test2, num_classes=3, num_epochs=500, alpha=0.005)

train_confusion_matrix = metrics.confusion_matrix(tr_targets2, tr_predictions2)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = train_confusion_matrix, display_labels = [0, 1, 2])
cm_display.plot()
plt.show()

#Confusion matrices and errror rate for training set
test_confusion_matrix = metrics.confusion_matrix(test_targets2, test_predictions2)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = test_confusion_matrix, display_labels = [0, 1, 2])
cm_display.plot()
plt.show()



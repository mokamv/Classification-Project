import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelBinarizer



with open('iris.data', 'r') as file: 
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
numeric_labels = lb.transform(y_labels)



#Splitting of data into a test set and a training set
X_train = np.array(x_data[0:30])
x_test = x_data[30:50] + x_data[80:100] + x_data[130:150]

T_train = np.concatenate(numeric_labels[0:30])
t_test = np.concatenate([numeric_labels[30:50], numeric_labels[80:100], numeric_labels[130:150]], axis=0)


#Initialization of weight matrix
D = 4 #Number of features
C = 3 #Number of classes

#We define the sigmoid funciton so the output of the classifier fits in the [0,1] range

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


#Computing the MSE gradient

def compute_gradients(X, G, T):
    delta = (G - T) * G * (1 - G)     # shape: (N, C)
    grad_W = delta.T @ X             # (C, N) @ (N, D+1) = (C, D+1)
    return grad_W


def train_linear_classifier(X, T, num_classes, num_epochs=1000, alpha=0.05):
    N, D_plus_1 = X.shape
    W = np.random.randn(num_classes, D_plus_1) * 0.01  # weight
    loss_history = []

    for epoch in range(num_epochs):

        Z = X_train @ W.T
        G = sigmoid(Z)
        loss = compute_mse(G, T)
        loss_history.append(loss)
        grad_W = compute_gradients(X, G, T)
        W -=alpha* grad_W
        print(f"Epoch {epoch}, Loss: {loss:.5f}")


    return W, loss_history

def compute_mse(G, T):
    return 0.5 * np.mean(np.sum((G - T)**2, axis=1))  # per-sample MSE



W_trained, loss_history = train_linear_classifier(X_train, T_train, num_classes=3, num_epochs=1000, alpha=0.05)
plt.plot(loss_history)
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Training Loss Over Time")
plt.ylim(min(loss_history)-0.01, max(loss_history)+0.01)  # zoom in
plt.grid(True)
plt.show()


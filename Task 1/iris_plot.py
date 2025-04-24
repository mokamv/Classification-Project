import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

# Open the file for reading 
with open('class_1', 'r') as file: 
    # Read the contents of the file 
    data = file.readlines()
 


my_data = [line.strip().split(',') for line in data]

data = [[float(x) for x in row[:4]] for row in my_data]


x = [row[0] for row in data]
y = [row[1] for row in data]

plt.scatter(x, y, color = "green")
plt.xlabel("Sepal length")
plt.ylabel("Sepal width")
plt.title("Iris-setosa Sepal vs Width")
plt.grid(True)
plt.show()
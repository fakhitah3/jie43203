import streamlit as st
# Import necessary modules
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

df = pd.read_csv("https://raw.githubusercontent.com/fakhitah3/jie43203/refs/heads/main/Iris.csv")

st.write(df)

# Loading data
irisData = load_iris()



# Create feature and target arrays
X = irisData.data
y = irisData.target

# Split into training and test set
X_train, X_test, y_train, y_test = train_test_split(
			X, y, test_size = 0.2, random_state=42)


X = df.iloc[:, 1:5].values
y = df['Species'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)

knn3 = KNeighborsClassifier(n_neighbors=3)
knn3.fit(X_train, y_train)

# Predict on dataset which model has not seen before
print(knn3.predict(X_test))

knn5 = KNeighborsClassifier(n_neighbors=7)
knn5.fit(X_train, y_train)

# Predict on dataset which model has not seen before
print(knn5.predict(X_test))

# Calculate the accuracy of the model
print("Accuracy n=3", knn3.score(X_test, y_test))
print("Accuracy n=5", knn5.score(X_test, y_test))


neighbors = np.arange(1, 9)
train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))

# Loop over K values
for i, k in enumerate(neighbors):
	knn = KNeighborsClassifier(n_neighbors=k)
	knn.fit(X_train, y_train)

	# Compute training and test data accuracy
	train_accuracy[i] = knn.score(X_train, y_train)
	test_accuracy[i] = knn.score(X_test, y_test)

# Generate plot
plt.plot(neighbors, test_accuracy, label = 'Testing dataset Accuracy')
plt.plot(neighbors, train_accuracy, label = 'Training dataset Accuracy')

plt.legend()
plt.xlabel('n_neighbors')
plt.ylabel('Accuracy')
plt.show()

st.pyplot(plt.gcf())

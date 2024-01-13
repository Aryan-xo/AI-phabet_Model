### Digit Recognition AI Model
This project implements a simple digit recognition AI model using the MNIST dataset. The model is implemented in Python with TensorFlow and NumPy. It includes the following steps:

# Installation and Dependencies
bash
Copy code
pip install pandas numpy matplotlib tensorflow
Loading and Preprocessing the Data
python
Copy code
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Load and preprocess the MNIST dataset
(X_train, Y_train), (X_test, Y_test) = tf.keras.datasets.mnist.load_data()
X_train = X_train.reshape(-1, 784).astype('float32') / 255.0
X_test = X_test.reshape(-1, 784).astype('float32') / 255.0
Y_train = Y_train.astype('int32')
Y_test = Y_test.astype('int32')

# Split the dataset into training and validation sets
from sklearn.model_selection import train_test_split
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.1, random_state=42)
Visualizing the Data
python
Copy code
import random

# Display a random sample from the dataset
index_to_display = random.randint(0, len(X_test) - 1)
sample_image = X_train[index_to_display].reshape(28, 28)

plt.imshow(sample_image, cmap='gray')
plt.title(f'Label: {Y_train[index_to_display]}')
plt.show()
Model Implementation
python
Copy code
# Model parameters
i = 784
h1 = 128
h2 = 64
o = 10
lr = 0.01
epochs = 15

# ... (code for initializing weights, biases, and activation functions)

# Training loop
for ep in range(epochs):
    # ... (training code)

    # Print results at the end of each epoch
    print(f"Epoch {ep + 1}/{epochs} - Average Cost: {total_cost / len(X_train):.4f}, Training Accuracy: {correct_predictions / len(X_train) * 100:.2f}%, Validation Accuracy: {accuracy_val * 100:.2f}%")
Testing the Model
python
Copy code
# Display an image from the X_test set
index_to_display = random.randint(0, len(X_test) - 1)
sample_image = X_test[index_to_display].reshape(28, 28)

plt.imshow(sample_image, cmap='gray')
plt.title(f"True Label: {Y_test[index_to_display]}")
plt.show()

# Use the classify function to get model's output
image_to_classify = X_test[index_to_display]
predictions = classify(image_to_classify, W1, b1, W2, b2, W3, b3)

# Determine the most likely predicted digit
predicted_digit = np.argmax(predictions)
print(f"Predicted Digit: {predicted_digit}")
Model Evaluation
python
Copy code
# Evaluate the model on the test set
correct_predictions = 0

for i in range(len(X_test)):
    image_to_classify = X_test[i]
    predictions = classify(image_to_classify, W1, b1, W2, b2, W3, b3)
    predicted_digit = np.argmax(predictions)

    if predicted_digit == Y_test[i]:
        correct_predictions += 1

accuracy = correct_predictions / len(X_test) * 100
print(f"Accuracy on Test Set: {accuracy:.2f}%")
Exporting and Loading Model Weights
python
Copy code
# Export weights and biases as numpy files
np.save('W1.npy', W1)
np.save('b1.npy', b1)
np.save('W2.npy', W2)
np.save('b2.npy', b2)
np.save('W3.npy', W3)
np.save('b3.npy', b3)

# Load weights and biases from numpy files
loaded_W1 = np.load('W1.npy')
loaded_b1 = np.load('b1.npy')
loaded_W2 = np.load('W2.npy')
loaded_b2 = np.load('b2.npy')
loaded_W3 = np.load('W3.npy')
loaded_b3 = np.load('b3.npy')
Feel free to customize the README according to your project details and structure. If you have additional sections or information, include them for a comprehensive README.

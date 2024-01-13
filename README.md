# Digit Recognition AI Model

## Overview
This project implements a simple digit recognition AI model using the MNIST dataset. The primary goal is to recognize handwritten digits from 0 to 9. The model is implemented in Python using TensorFlow and NumPy.

## Dataset
The MNIST dataset consists of 28x28 pixel grayscale images of handwritten digits. It is a widely used dataset for training and testing image processing systems.

## Model Architecture
The model consists of three layers:

1. **Input Layer:** 784 neurons (28x28 pixels) representing each pixel in the input image.
2. **Hidden Layer 1:** 128 neurons with a sigmoid activation function.
3. **Hidden Layer 2:** 64 neurons with a sigmoid activation function.
4. **Output Layer:** 10 neurons corresponding to the digits 0 through 9, with a sigmoid activation function.

## Mathematics Behind the Model
### Forward Propagation
1. The input image is flattened into a vector of 784 values.
2. The dot product of the input vector and the weights of the first layer is computed, and the bias is added.
3. The sigmoid activation function is applied to the result.
4. Steps 2 and 3 are repeated for the subsequent hidden layers and the output layer.

### Back Propagation
1. The predicted output is compared to the actual label using the cross-entropy loss.
2. The gradients with respect to the weights and biases are calculated using the chain rule.
3. The weights and biases are updated using gradient descent to minimize the loss.

## Training
The model is trained using the MNIST training dataset, and its performance is evaluated on a validation set. The training loop runs for a specified number of epochs, adjusting the weights and biases to improve accuracy.

## Testing
The model is tested on a separate test set to evaluate its generalization performance. The accuracy on the test set provides insights into how well the model performs on unseen data.

## Results
The training loop prints the average cost (cross-entropy loss), training accuracy, and validation accuracy at the end of each epoch. These metrics help monitor the model's progress during training.

## Exporting and Loading Model Weights
The weights and biases learned during training can be exported to numpy files for later use. This allows the model to be loaded with pre-trained parameters for quick predictions or further fine-tuning.

## Conclusion
This digit recognition AI model showcases the fundamentals of neural networks and provides a practical example of image classification. The mathematics involved in forward and backward propagation are essential components of understanding how neural networks learn from data.

Feel free to explore the code and adapt the model for your specific use cases or datasets. The provided Python code demonstrates the implementation details of the model, and you can further customize it based on your requirements.

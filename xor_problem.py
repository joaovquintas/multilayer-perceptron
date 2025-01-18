import numpy as np

def sigmoid(x):
    return 1/ (1 + np.exp(-x))
def sig_derivative(sigmoid):
    return sigmoid * (1 - sigmoid)

### Cost function - Binary Cross-Entropy Loss
def binary_cross_entropy(y_true, y_predict):
    epsilon = 1e-15
    y_predict = np.clip(y_predict, epsilon, 1 - epsilon)  # Restringe valores para evitar erros
    bce = -np.mean(y_true * np.log(y_predict) + (1 - y_true) * np.log(1 - y_predict))
    return bce

def binary_cross_entropy_derivative(y_true, y_predict):
    epsilon = 1e-15
    y_predict = np.clip(y_predict, epsilon, 1 - epsilon)
    return -(y_true / y_predict) + (1 - y_true) / (1 - y_predict)

## Input and Output
X = np.array([[0,0],
              [0,1],
              [1,0],
              [1,1]])

Y = np.array([[0], 
              [1], 
              [1], 
              [0]])


## Weights, biases and hidden layer's dimensions
input_dimension = 2
hidden_layer_dim = 2
output_layer_dim = 1
batch_size, features = X.shape
W1 = np.random.random((input_dimension, hidden_layer_dim))
b1 = np.random.random((1, hidden_layer_dim))
W2 = np.random.random((hidden_layer_dim, 1))
b2 = np.random.random((1, output_layer_dim))


## Trainning
epochs = 30000
learning_rate = 0.1

for epoch in range(epochs):

    #FEEDFOWARD PASS

    Z1 = np.dot(X, W1) + b1
    A1 = sigmoid(Z1)
    Z2 = np.dot(A1, W2) + b2
    A2 = sigmoid(Z2)

    loss = binary_cross_entropy(Y, A2)
    if epoch % 1000 == 0:
        print(f"Epoch: {epoch}, Loss: {loss:.4f}")

    #BACKPROPAGATION 

    ## OUTPUT GRADIENT
    error_output = binary_cross_entropy_derivative(Y, A2)
    delta_error_output = error_output * sig_derivative(A2)


    ## HIDDEN GRADIENT
    error_hidden = np.dot(delta_error_output, W2.T)

    delta_error_hidden = error_hidden * sig_derivative(A1)

    ## GRADIENT DESCENT

    W2 -= np.dot(A1.T, delta_error_output) * learning_rate
    b2 -= np.sum(delta_error_output, keepdims=True, axis=0)
    W1 -= np.dot(X.T, delta_error_hidden) *learning_rate
    b1 -= np.sum(delta_error_hidden, keepdims=True, axis=0)

print(A2)


import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from keras.datasets import mnist

# Load MNIST dataset
def load_mnist():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    # Normalize pixel values to between 0 and 1
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0
    # Flatten images into 1D vectors
    X_train = X_train.reshape((X_train.shape[0], -1))
    X_test = X_test.reshape((X_test.shape[0], -1))
    # Convert labels to categorical (one-hot encoding)
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    return X_train, y_train, X_test, y_test

# Initialize parameters (weights and biases)
def initialize_parameters(input_size, hidden_size, output_size):
    np.random.seed(42)
    W1 = np.random.randn(input_size, hidden_size) * 0.01
    b1 = np.zeros((1, hidden_size))
    W2 = np.random.randn(hidden_size, output_size) * 0.01
    b2 = np.zeros((1, output_size))
    return W1, b1, W2, b2

def relu(Z):
    return np.maximum(0,Z)
    
def relu_derivates(Z): #we need relu derivation for backprop 
    return (Z>0).astype(int)
'''
- returns true if Z is greater than 0, ese returns false
- converts true and false to 1 and 0
'''

#softmax, it might give higher numbers as there is e^z
def softmax(z):
    exp_z = np.exp(z.astype(float))+1e-15
    return exp_z/np.sum(exp_z)


# Forward propagation
def forward_propagation(X,parameters): 
    W1, b1, W2, b2 = parameters
    # Hidden layer
    X = X.astype('float32')
     
    Z1 = np.dot(X,W1) + b1
    A1 = relu(Z1)
    
    # Output layer
    Z2 = np.dot(A1,W2) + b2
    A2 = softmax(Z2)
    
    cache = {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}
    return A2,cache
# Compute cross-entropy loss


def compute_cost(y, y_hat):
    y_hat = y_hat + 1e-15#to avoid log(0)
    loss = - np.sum(y*np.log(y_hat))
    cost = np.mean(loss)
    return cost

def backward_propagation(X,Y,parameters,cache):
    m = X_train.shape[0]
    A1,A2 = cache['A1'],cache['A2']
    W1, b1, W2, b2 = parameters
    
    dZ2= A2 - Y
    dW2 = (1/m)*np.dot(A1.T,dZ2)
    db2 = (1/m)*np.sum(dZ2,axis=0,keepdims=True)
    
    dZ1 = np.dot(dZ2,W2.T)*relu_derivates(A1)
    dW1 = (1/m)*np.dot(X.T,dZ1)
    db1 = (1/m)*np.sum(dZ1,axis=0,keepdims=True)
    
    grads = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}
    
    return grads

def update_paramters(parameters,grads,learning_rate):
    W1, b1, W2, b2 = parameters
    dW1, db1, dW2, db2 = grads['dW1'], grads['db1'], grads['dW2'], grads['db2']
    
    W1 = W1 - learning_rate*dW1
    b1 = b1 - learning_rate*db1
    W2 = W2 - learning_rate*dW2
    b2 = b2 - learning_rate*db2
    
    parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}
    
    return parameters

def accuracy(y, y_hat):
    return np.mean(y == y_hat)

def train_model(X,input_size,hidden_size,output_size,y,learning_rate,epochs):
    
    parameters = initialize_parameters(input_size, hidden_size, output_size)
    
    for i in range(epochs):
        y_hat,cache = forward_propagation(X,parameters)
        cost = compute_cost(y, y_hat)
        grads = backward_propagation(X,y,parameters,cache)
        parameters = update_paramters(parameters,grads,learning_rate)
        
        acc = accuracy(y, y_hat)
        if i % 100 == 0:
            print(f'Epoch {i}, cost: {cost}- accuracy: {acc}')
    return parameters,cost,acc

X_train, y_train, X_test, y_test = load_mnist()
parameters,cost,accuracy = train_model(X_train,784,64,10,y_train,0.01,1000)
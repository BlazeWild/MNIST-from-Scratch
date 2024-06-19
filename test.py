import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from keras.datasets import mnist


def load_data():
    (X_train,y_train), (X_test, y_test) = mnist.load_data()
    
    #normalize data as original input ranges from 0 to 255
    X_train = X_train/255
    X_train = X_train/255
    
    #THe mnist model is :
    
    X_train.reshape(X_train.shape[0],-1)
    X_test.reshape(X_test.shape[0],-1)
    
    #one hot encode the output
    num_classes = 10 #10 classes
    y_train = to_categorical(y_train,num_classes)
    y_test = to_categorical(y_test,num_classes)
    
    return X_train,y_train,X_test,y_test

X_train,y_train, X_test, y_test = load_data()

def initialize_parameters(input_size, hidden_size,output_size):
    np.random.seed(42)
    #randn for initialize randomly  with numbers having mean 0 and variance 1( for better symmeteric learning)
    W1 = np.random.randn(input_size,hidden_size) * 0.01 
    b1 = np.zeros((hidden_size,1))
    W2 = np.random.randn(hidden_size,output_size) * 0.01 
    b2 = np.zeros((output_size,1))
    return W1,b1,W2,b2


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
    exp_z = np.exp(z)
    return exp_z/np.sum(exp_z)

def forward_prop(X,W1,b1,W2,b2):
    Z1 = np.dot(W1.T,X)+b1
    A1 = relu(Z1)
    Z2 = np.dot(W2.T,A1)+b2
    A2 = softmax(Z2)
    
    return Z1,A1,Z2,A2

def compute_crossentropy_loss(y,y_hat): # this is where the softmax is applied, ie in the output layer
    #for loss/error
    y_hat = y_hat + 1e-15#to avoid log(0)
    loss = - np.sum(y*np.log(y_hat))
    return loss

def compute_cost(y,y_hat):
    losses = compute_crossentropy_loss(y,y_hat)
    cost = np.mean(losses) #cost is the mesn of losses
    return cost

def back_prop(X,A2,A1,Z1,W1,W2,Y):

    m = X.shape[0] #number of training examples 
    
    dZ2 = A2-Y
    dW2 = np.matmul(A1,dZ2)/m
    db2 = (1/m)* np.sum(dZ2,1) #during dZ2 we already did 1/m, so we need to sum for this
    
    dA1 = np.matmul(W2,dZ2)
    dZ1 = dA1 * relu_derivates(Z1)
    dW1 = (1/m) * np.matmul(X,dZ1.T)
    db1 = (1/m) * np.sum(dZ1,1)
    
    return dW2,db2,dW1,db1

def update_parameters(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate):
    W1 -= learning_rate*dW1
    b1 -= learning_rate*db1
    W2 -= learning_rate*dW2
    b2 -= learning_rate*db2
    
    return W1,b1,W2,b2

def calculate_accuracy(y,y_hat):
    predictions = np.argmax(y_hat,axis = 0) #returns index with highest y_hat
    labels = np.argmax(y,axis=0)
    accuracy = np.mean(predictions == labels)
    
    return accuracy

def train_model(X_train,y_train,X_test,y_test,input_size,hidden_size,output_size,epochs,learning_rate):
    
    W1,b1,W2,b2 = initialize_parameters(input_size, hidden_size,output_size)
    
    for epoch in range(epochs):
        #forward prop
        Z1,A1,Z2,A2 = forward_prop(X_train,W1,b1,W2,b2)
        
        #compute cost
        cost = compute_cost(y_train,A2)
        
        #back prop
        dW2,db2,dW1,db1 = back_prop(X_train,A2,A1,Z1,W1,W2,y_train)
        
        #update param
        W1,b1,W2,b2 = update_parameters(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate)
        
        #show cost for each 10 iterations/epoch
        if epoch % 10 == 0:
            train_accuracy = calculate_accuracy(y_train, A2)
            print(f"Epoch {epoch}/{epochs} - Cost : {cost:.4f} - Training Accuracy: {train_accuracy}")
        
        
    #calculate on test set
    _,_,_,A2_test = forward_prop(X_test,W1,b1,W2,b2) #The _ placeholders are used for intermediate values (Z1, A1, Z2) that are not needed for this final evaluation step
    test_cost = compute_cost(y_test, A2_test)
    print(f"Final test cost : {test_cost}")
    
    return W1,b1,W2,b2,test_cost,train_accuracy
  
  #load dataset
X_train, y_train, X_test, y_test = load_data()
##Initialize parameters
input_size =# this is 784, as a picture is a 28*28 pixels
hidden_size = 64
output_size = 10 #10 numbers to classify
epochs = 100
learning_rate = 0.01


W1,b1,W2,b2,test_cost,train_accuracy = train_model(X_train, y_train, X_test, y_test, 
                                            input_size, hidden_size,output_size, epochs, learning_rate)
import random
import numpy as np

def sigmoid(x): 
    return 1 / (1 + np.exp(-x))

def sigmoid_backward(sx):
    return sx * (1 - sx)

def build_xor(z):
    n = [0, 0, 0], [1, 0, 1], [0, 1, 1], [1, 1, 0]
    
    xor = []
    for _ in range(z):
        idx = random.randint(0,3)
        xor.append(n[idx])
        
    return np.array(xor).flatten()

xor = build_xor(10)

encoding_dim = 1
hidden_dim = 2
output_dim = 1

w1 = np.random.random((hidden_dim + encoding_dim, hidden_dim))
w2 = np.random.random((hidden_dim, output_dim))

b1 = np.random.random((1, hidden_dim))
b2 = np.random.random((1, output_dim))

context = np.zeros([1, hidden_dim + encoding_dim]) + 0.5

epochs = len(xor)-1
lr = 0.3

for i in range(epochs):

    context[:,2] = xor[i]

    print(context)

    hidden = sigmoid(context @ w1 + b1)
    logits = sigmoid(hidden @ w2 + b2)

    context[:,0] = hidden.squeeze()[0]
    context[:,1] = hidden.squeeze()[1]

    y = np.array([xor[i+1]])

    output_error = logits - y
    dL_dy = output_error * sigmoid_backward(logits)

    hidden_error = dL_dy @ w2.T
    dL_dw = hidden_error * sigmoid_backward(hidden)


    w2 -= lr * (hidden.T @ dL_dy)
    b2 -= lr * np.sum(dL_dy, axis=0, keepdims=True)
    
    w1 -= lr * (context.T @ dL_dw)
    b1 -= lr * np.sum(dL_dw, axis=0, keepdims=True)
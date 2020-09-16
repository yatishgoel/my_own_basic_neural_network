import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_derivative(x):
    return x*(1-x)


training_input = np.array([[0,0,1],[1,1,1],[1,0,1],[0,1,1]])

training_output = np.array([[0,1,1,0]]).T

np.random.seed(1)

synaptic_weighs = 2*np.random.random((3,1))-1

print('random weights :')
print(synaptic_weighs)

for iteration in range(500000):
    input_layer = training_input

    outputs = sigmoid(np.dot(input_layer, synaptic_weighs))

    error = training_output - outputs

    adjustments = error * sigmoid_derivative(outputs)

    synaptic_weighs += np.dot(input_layer.T,adjustments)


print('outputs')
print(outputs)

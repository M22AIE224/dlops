import numpy as np
import matplotlib.pyplot as plt

def sigmoid(input_val):
    sig_value = 1 / (1 + np.exp(-input_val))
    return sig_value


def relu(input_val):
    relu_val = np.maximum(0, input_val)
    return relu_val

def leaky_relu(input_val, alpha=0.01):
    leaky_relu_val = np.where(input_val > 0, input_val, alpha * input_val)

    return leaky_relu_val

def tanh(input_val):
    tanh_val = np.tanh(input_val)
    return tanh_val

# Generate data
input_val = np.linspace(-5, 5, 100)

print(input_val)
sig_val = sigmoid(input_val)
relu_val = relu(input_val)
leaky_relu_val = leaky_relu(input_val)
tanh_val = tanh(input_val)

# Plotting
plt.figure(figsize=(20, 6))

plt.plot(input_val, sig_val, label='Sigmoid')
plt.plot(input_val, relu_val, label='ReLU')
plt.plot(input_val, leaky_relu_val, label='Leaky ReLU')
plt.plot(input_val, tanh_val, label='Tanh')

plt.title('Activation Functions')
plt.xlabel('Input')
plt.ylabel('Output')
plt.legend()
plt.grid(True)
plt.show()

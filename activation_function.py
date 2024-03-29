# Feature 1 Sigmoid for list of data
import numpy as np

#print(np.linspace(-5, 5, 100))

random_values = np.array([-3.5, -1.2, 0, 2.8, -4.1, 1.5, -0.7, 3.2, -2.4, 4.6])
print("Original Input Data - ", np.array(random_values))

#def getSigmoid(input_val):
#    sig_value = 1 / (1 + np.exp(-input_val))
#    return sig_value

#sig_output_val = getSigmoid(random_values)

#print("Sigmoid Value of Input Data - ", sig_output_val)

def getRelu(input_val):
    relu_val = np.maximum(0, input_val)
    return relu_val

def getLeaky_relu(input_val, alpha=0.01):
    leaky_relu_val = np.where(input_val > 0, input_val, alpha * input_val)

    return leaky_relu_val

def getTanh(input_val):
    tanh_val = np.tanh(input_val)
    return tanh_val   


relu_output_val = getRelu(random_values)
leakyrelu_output_val = getLeaky_relu(random_values)
tanh_output_val = getTanh(random_values)


print("Relu Value of Input Data - ", relu_output_val)
print("Leaky Relu Value of Input Data - ", leakyrelu_output_val)
print("Tanh Value of Input Data - ", tanh_output_val)
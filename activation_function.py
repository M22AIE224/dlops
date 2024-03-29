# Feature 1 Sigmoid for list of data
import numpy as np

#print(np.linspace(-5, 5, 100))

random_values = np.array([-3.5, -1.2, 0, 2.8, -4.1, 1.5, -0.7, 3.2, -2.4, 4.6])
print("Original Input Data - ", np.array(random_values))

def getSigmoid(input_val):
    sig_value = 1 / (1 + np.exp(-input_val))
    return sig_value

sig_output_val = getSigmoid(random_values)

print("Sigmoid Value of Input Data - ", sig_output_val)

    
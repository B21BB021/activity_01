import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)
  
def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

def tanh(x):
    return np.tanh(x)

# Generate data
x = np.linspace(-5, 5, 100)
y_sigmoid = sigmoid(x)
y_relu = relu(x)
y_leaky_relu = leaky_relu(x)
y_tanh = tanh(x)

# Plot graphs
plt.figure(figsize=(10, 6))

plt.subplot(2, 2, 1)
plt.plot(x, y_sigmoid, label='Sigmoid')
plt.title('Sigmoid')
plt.xlabel('x')
plt.ylabel('sigmoid(x)')
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(x, y_relu, label='ReLU')
plt.title('ReLU')
plt.xlabel('x')
plt.ylabel('relu(x)')
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(x, y_leaky_relu, label='Leaky ReLU')
plt.title('Leaky ReLU')
plt.xlabel('x')
plt.ylabel('leaky_relu(x)')
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(x, y_tanh, label='Tanh')
plt.title('Tanh')
plt.xlabel('x')
plt.ylabel('tanh(x)')
plt.legend()

plt.tight_layout()
plt.show()

import numpy as np
import matplotlib.pyplot as plt

# Define the objective function and its gradient
def f(x):
    return x**2  # Objective function f(x) = x^2

def grad_f(x):
    return 2*x  # Gradient of the function ∇f(x) = 2x

# Gradient descent algorithm
def gradient_descent(learning_rate=0.1, iterations=20):
    x = 10  # Initial point
    x_values = [x]  # Store x values for visualization

    for _ in range(iterations):
        x -= learning_rate * grad_f(x)  # Update x in the negative gradient direction
        x_values.append(x)  # Store updated x values

    return x_values

# Run the gradient descent algorithm
x_vals = gradient_descent()

# Plot the function and optimization process
x = np.linspace(-10, 10, 400)  # Generate x values from -10 to 10
y = f(x)  # Compute corresponding y values

plt.plot(x, y, label="f(x) = x^2")  # Plot the function
plt.scatter(x_vals, [f(x) for x in x_vals], color='red', label="Descent Steps")  # Plot descent steps
plt.xlabel("x")
plt.ylabel("f(x)")
plt.legend()
plt.title("Gradient Descent Optimization")
plt.show()

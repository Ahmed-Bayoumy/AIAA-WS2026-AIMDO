import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt

# 1. Define the Computational Domain
# We'll work on the interval [0, 1]
geom = dde.geometry.Interval(0, 1)

# 2. Define the Governing Equation (PDE/ODE)
# Our constraint is du/dx - 2x = 0
# In DeepXDE, we define the PDE residual function.
# y is the output of the neural network (our u(x))
# x is the input (our x coordinate)
def pde(x, y):
    # y is a tensor of shape (N, 1) where N is the number of points
    # We need the first derivative of y with respect to x
    dy_dx = dde.grad.jacobian(y, x, i=0, j=0)
    # The PDE residual is dy/dx - 2x
    return dy_dx - 2 * x

# 3. Define Boundary Conditions (BCs)
# We have u(0) = 0
# This is a Dirichlet boundary condition.
# The 'on_boundary' function checks if a point is on the boundary.
def boundary_condition(x, on_boundary):
    return on_boundary and np.isclose(x[0], 0)

# Define the DirichletBC object
bc = dde.DirichletBC(geom, lambda x: 0, boundary_condition)

# 4. Define the Objective Function (Implicitly handled by PDE/BC loss)
# For this simple problem, minimizing the integral J = integral((u(x) - x^2)^2) dx
# is equivalent to finding u(x) = x^2 that satisfies the PDE and BC.
# The PINN will inherently try to find u(x) that satisfies the PDE and BCs,
# which for this specific problem, is the solution that also minimizes J.
# If the objective were more complex or decoupled, we might add a custom loss term.

# 5. Create the Neural Network Model
# Define the number of input and output dimensions
# Input: x (1D)
# Output: u(x) (1D)
num_input = 1
num_output = 1
num_hidden_layers = 3
num_neurons_per_layer = 20

# Define the neural network architecture
# We'll use a fully connected neural network (FNN)
net = dde.maps.FNN([num_input] + [num_neurons_per_layer] * num_hidden_layers + [num_output],
                   "tanh",  # Activation function
                   "Glorot uniform" # Weight initializer
                  )

# 6. Train the PINN
# Combine the geometry, PDE, and boundary conditions into a DeepXDE data object
data = dde.data.PDE(geom, pde, bc, num_domain=2500, num_boundary=2, num_test=500)
# num_domain: Number of training points sampled in the domain
# num_boundary: Number of training points sampled on the boundary
# num_test: Number of test points for evaluation

# Create the DeepXDE model
model = dde.Model(data, net)

# Choose the optimizer and learning rate
# Adam is a good general-purpose optimizer
model.compile("adam", lr=0.001)

# Train the model
# iterations: Number of training epochs
# We can also use L-BFGS-B for fine-tuning after Adam
losshistory, train_state = model.train(iterations=10000)

# Optionally, train with L-BFGS-B for better convergence
# model.compile("L-BFGS-B")
# losshistory, train_state = model.train()

# 7. Evaluate the Results
# Generate test points
x_test = np.linspace(0, 1, 100)[:, np.newaxis] # Reshape for DeepXDE input

# Predict the solution u(x) using the trained model
u_pred = model.predict(x_test)

# Analytical solution: u(x) = x^2
u_analytical = x_test**2

# Plotting the results
plt.figure(figsize=(10, 6))
plt.plot(x_test, u_analytical, label="Analytical Solution: $u(x) = x^2$", linestyle='--', color='red')
plt.plot(x_test, u_pred, label="PINN Prediction", color='blue', alpha=0.7)
plt.xlabel("$x$")
plt.ylabel("$u(x)$")
plt.title("PINN Solution vs. Analytical Solution for Constrained Optimization BM Problem")
plt.legend()
plt.grid(True)
plt.show()

# Plot the loss history
dde.saveplot(losshistory, train_state, issave=False, isplot=True)

# Calculate and print the L2 relative error
l2_error = dde.metrics.l2_relative_error(u_analytical, u_pred)
print(f"L2 Relative Error: {l2_error:.4e}")
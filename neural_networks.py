import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.animation import FuncAnimation
import os
from functools import partial
from matplotlib.patches import Circle

result_dir = "results"
os.makedirs(result_dir, exist_ok=True)

# Define a simple MLP class
class MLP:
    def __init__(self, input_dim, hidden_dim, output_dim, lr, activation='tanh'):
        np.random.seed(0)
        self.lr = lr  # learning rate
        self.activation_fn = activation  # activation function

        # Initialize weights and biases with proper scaling
        self.W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2 / input_dim)
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, output_dim) * np.sqrt(2 / hidden_dim)
        self.b2 = np.zeros((1, output_dim))

        # Store intermediate values for visualization
        self.hidden_output = None
        self.input_gradients = None
        self.hidden_gradients = None

    def _activate(self, Z):
        """Applies the activation function."""
        if self.activation_fn == 'tanh':
            Z = np.clip(Z, -10, 10)  # Clip inputs to avoid overflow
            return np.tanh(Z), 1 - np.tanh(Z) ** 2
        elif self.activation_fn == 'relu':
            return np.maximum(0, Z), (Z > 0).astype(float)
        elif self.activation_fn == 'sigmoid':
            Z = np.clip(Z, -10, 10)  # Clip inputs to avoid overflow
            A = 1 / (1 + np.exp(-Z))
            return A, A * (1 - A)
        else:
            raise ValueError("Unsupported activation function!")

    def forward(self, X):
        """Performs forward propagation."""
        # Hidden layer
        Z1 = X @ self.W1 + self.b1
        A1, self.hidden_derivative = self._activate(Z1)
        self.hidden_output = A1  # For visualization

        # Output layer (no activation)
        Z2 = A1 @ self.W2 + self.b2
        return Z2

    def backward(self, X, y):
        """Computes gradients and updates weights."""
        # Forward pass to get outputs
        Z1 = X @ self.W1 + self.b1
        A1, _ = self._activate(Z1)
        Z2 = A1 @ self.W2 + self.b2
        preds = Z2

        # Compute the loss gradient (mean squared error)
        dZ2 = preds - y  # Output layer gradient
        dW2 = A1.T @ dZ2  # Weight gradient for W2
        db2 = np.sum(dZ2, axis=0, keepdims=True)  # Bias gradient for b2

        # Backpropagate through the hidden layer
        dA1 = dZ2 @ self.W2.T  # Gradient w.r.t A1
        dZ1 = dA1 * self.hidden_derivative  # Apply activation derivative
        dW1 = X.T @ dZ1  # Weight gradient for W1
        db1 = np.sum(dZ1, axis=0, keepdims=True)  # Bias gradient for b1

        # Gradient clipping to prevent instability
        dW1 = np.clip(dW1, -1.0, 1.0)
        db1 = np.clip(db1, -1.0, 1.0)
        dW2 = np.clip(dW2, -1.0, 1.0)
        db2 = np.clip(db2, -1.0, 1.0)

        # Add L2 regularization
        lambda_reg = 0.01
        dW1 += lambda_reg * self.W1
        dW2 += lambda_reg * self.W2

        # Update weights and biases
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

        # Store gradients for visualization
        self.input_gradients = np.abs(dW1)  # Magnitude of input-to-hidden gradients
        self.hidden_gradients = np.linalg.norm(dW2, axis=0)  # Magnitude of hidden-to-output gradients


def generate_data(n_samples=100):
    np.random.seed(0)
    # Generate input
    X = np.random.randn(n_samples, 2)
    y = (X[:, 0] ** 2 + X[:, 1] ** 2 > 1).astype(int) * 2 - 1  # Circular boundary
    y = y.reshape(-1, 1)
    return X, y

# Visualization update function
def update(frame, mlp, ax_input, ax_hidden, ax_gradient, X, y):
    ax_hidden.clear()
    ax_input.clear()
    ax_gradient.clear()

    # Perform 10 training steps
    for _ in range(10):
        mlp.forward(X)
        mlp.backward(X, y)

    # Plot hidden features
    hidden_features = mlp.hidden_output
    ax_hidden.scatter(hidden_features[:, 0], hidden_features[:, 1], c=y.ravel(), cmap='bwr', alpha=0.7)
    ax_hidden.set_title("Hidden Space at Step {}".format(frame * 10))

    # Visualize input space
    ax_input.scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap='bwr', alpha=0.7)
    ax_input.set_title("Input Space at Step {}".format(frame * 10))

    # Visualize gradients
    for i in range(mlp.W1.shape[0]):
        for j in range(mlp.W1.shape[1]):
            radius = mlp.input_gradients[i, j]
            circle = Circle((i, j + mlp.W1.shape[0]), radius=radius, color='purple', alpha=0.5)
            ax_gradient.add_patch(circle)

    ax_gradient.set_title("Gradients at Step {}".format(frame * 10))


def visualize(activation, lr, step_num):
    X, y = generate_data()
    mlp = MLP(input_dim=2, hidden_dim=3, output_dim=1, lr=lr, activation=activation)

    # Set up visualization
    fig = plt.figure(figsize=(21, 7))
    ax_hidden = fig.add_subplot(131, projection='3d')
    ax_input = fig.add_subplot(132)
    ax_gradient = fig.add_subplot(133)

    # Create animation
    ani = FuncAnimation(
        fig,
        partial(update, mlp=mlp, ax_input=ax_input, ax_hidden=ax_hidden, ax_gradient=ax_gradient, X=X, y=y),
        frames=step_num // 10,
        init_func=lambda: None,  # Add an empty init function if needed
        repeat=False
    )

    # Save the animation as a GIF
    ani.save(os.path.join(result_dir, "visualize.gif"), writer='pillow', fps=10)
    plt.close()


if __name__ == "__main__":
    activation = "tanh"
    lr = 0.1
    step_num = 1000
    visualize(activation, lr, step_num)


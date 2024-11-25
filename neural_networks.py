import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.animation import FuncAnimation
import os
from functools import partial
from matplotlib.patches import Circle

result_dir = "results"
os.makedirs(result_dir, exist_ok=True)

# Define the MLP class with fixed gradient handling
class MLP:
    def __init__(self, input_dim, hidden_dim, output_dim, lr, activation='tanh'):
        np.random.seed(0)
        self.lr = lr  # Learning rate
        self.activation_fn = activation  # Activation function
        # Initialize weights
        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.01
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, output_dim) * 0.01
        self.b2 = np.zeros((1, output_dim))

        # Set activation and its derivative
        if activation == 'tanh':
            self.activation = np.tanh
            self.activation_derivative = lambda z: 1 - np.tanh(z) ** 2
        elif activation == 'relu':
            self.activation = lambda z: np.maximum(0, z)
            self.activation_derivative = lambda z: (z > 0).astype(float)
        elif activation == 'sigmoid':
            self.activation = lambda z: 1 / (1 + np.exp(-z))
            self.activation_derivative = lambda z: self.activation(z) * (1 - self.activation(z))
        else:
            raise ValueError("Unsupported activation function.")

        self.gradients = {}  # To store gradients for visualization

    def forward(self, X):
        self.Z1 = X @ self.W1 + self.b1
        self.A1 = self.activation(self.Z1)
        self.Z2 = self.A1 @ self.W2 + self.b2
        self.A2 = self.activation(self.Z2)
        return self.A2


    def backward(self, X, y):
        # Compute gradients
        m = X.shape[0]
        dZ2 = (self.A2 - y) / m
        dW2 = self.A1.T @ dZ2
        db2 = np.sum(dZ2, axis=0, keepdims=True)

        dA1 = dZ2 @ self.W2.T
        dZ1 = dA1 * self.activation_derivative(self.Z1)
        dW1 = X.T @ dZ1
        db1 = np.sum(dZ1, axis=0, keepdims=True)

        # Store gradients for visualization
        self.gradients = {'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2}

        # Gradient descent step
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2

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

    # Perform multiple training steps
    for _ in range(10):
        mlp.forward(X)
        mlp.backward(X, y)

    # Plot hidden features
    hidden_features = mlp.A1
    if hidden_features.shape[1] == 2:
        ax_hidden.scatter(hidden_features[:, 0], hidden_features[:, 1], c=y.ravel(), cmap='bwr', alpha=0.7)
        ax_hidden.set_title("Hidden Space")
        ax_hidden.set_xlabel("Hidden Feature 1")
        ax_hidden.set_ylabel("Hidden Feature 2")
    elif hidden_features.shape[1] == 3:
        ax_hidden.scatter(hidden_features[:, 0], hidden_features[:, 1], hidden_features[:, 2], c=y.ravel(), cmap='bwr', alpha=0.7)
        ax_hidden.set_title("Hidden Features")
        ax_hidden.set_xlabel("Hidden Feature 1")
        ax_hidden.set_ylabel("Hidden Feature 2")
        ax_hidden.set_zlabel("Hidden Feature 3")

    # Distorted input space transformed by the hidden layer
    ax_input.scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap='bwr', alpha=0.7)
    ax_input.set_title("Input Space")
    ax_input.set_xlabel("Input Feature 1")
    ax_input.set_ylabel("Input Feature 2")

    # Plot decision boundary
    xx, yy = np.meshgrid(np.linspace(-3, 3, 100), np.linspace(-3, 3, 100))
    grid = np.c_[xx.ravel(), yy.ravel()]
    preds = mlp.forward(grid).ravel()
    preds = preds.reshape(xx.shape)
    ax_input.contourf(xx, yy, preds, levels=50, cmap='bwr', alpha=0.3)

    # Gradient visualization with labels
    ax_gradient.clear()

    # Input, hidden, and output positions
    input_positions = [(0, 0), (0, 1)]  # For x1 and x2
    hidden_positions = [(0.5, 0), (0.5, 0.5), (0.5, 1)]  # For h1, h2, h3
    output_position = (1, 0.5)  # For y

    # Add neurons
    ax_gradient.scatter(*zip(*input_positions), c='blue', s=200, label='Input')  # Input neurons
    ax_gradient.scatter(*zip(*hidden_positions), c='blue', s=200, label='Hidden')  # Hidden neurons
    ax_gradient.scatter(*output_position, c='blue', s=200, label='Output')  # Output neuron

    # Add labels
    ax_gradient.text(0, 0, 'x1', ha='center', va='center', fontsize=12, color='white')
    ax_gradient.text(0, 1, 'x2', ha='center', va='center', fontsize=12, color='white')
    ax_gradient.text(0.5, 0, 'h1', ha='center', va='center', fontsize=12, color='white')
    ax_gradient.text(0.5, 0.5, 'h2', ha='center', va='center', fontsize=12, color='white')
    ax_gradient.text(0.5, 1, 'h3', ha='center', va='center', fontsize=12, color='white')
    ax_gradient.text(1, 0.5, 'y', ha='center', va='center', fontsize=12, color='white')

    # Visualize connections with gradients
    max_grad = max(np.abs(mlp.gradients['W1']).max(), np.abs(mlp.gradients['W2']).max())
    max_grad = max_grad if max_grad != 0 else 1

    # Input to hidden connections
    for i, (x, y) in enumerate(input_positions):
        for j, (hx, hy) in enumerate(hidden_positions):
            gradient = mlp.gradients['W1'][i, j]
            thickness = 5 * abs(gradient) / max_grad
            ax_gradient.plot(
                [x, hx], [y, hy],
                linewidth=thickness,
                color="purple" if gradient > 0 else "pink",
                alpha=0.7,
            )

    # Hidden to output connections
    for i, (hx, hy) in enumerate(hidden_positions):
        gradient = mlp.gradients['W2'][i, 0]
        thickness = 5 * abs(gradient) / max_grad
        ax_gradient.plot(
            [hx, output_position[0]], [hy, output_position[1]],
            linewidth=thickness,
            color="purple" if gradient > 0 else "pink",
            alpha=0.7,
        )

    # Set limits and title
    ax_gradient.set_xlim(-0.2, 1.2)
    ax_gradient.set_ylim(-0.2, 1.2)
    ax_gradient.set_title(f"Gradients")



def visualize(activation, lr, step_num):
    X, y = generate_data()
    mlp = MLP(input_dim=2, hidden_dim=3, output_dim=1, lr=lr, activation=activation)

    # Set up visualization
    matplotlib.use('agg')
    fig = plt.figure(figsize=(30, 10))
    ax_hidden = fig.add_subplot(131, projection='3d')
    ax_input = fig.add_subplot(132)
    ax_gradient = fig.add_subplot(133)

    # Create animation
    ani = FuncAnimation(fig, partial(update, mlp=mlp, ax_input=ax_input, ax_hidden=ax_hidden, ax_gradient=ax_gradient, X=X, y=y), frames=step_num//10, repeat=False)

    # Save the animation as a GIF
    ani.save(os.path.join(result_dir, "visualize.gif"), writer='pillow', fps=10)
    plt.close()

if __name__ == "__main__":
    activation = "tanh"
    lr = 0.1
    step_num = 1000
    visualize(activation, lr, step_num)
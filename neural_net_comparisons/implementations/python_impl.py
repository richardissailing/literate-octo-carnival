import math
import random
from typing import List, Dict, Optional
from dataclasses import dataclass
import time
from tqdm import tqdm
import numpy as np
from .matrix import Matrix
from .activations import ActivationFunctions

@dataclass
class LayerConfig:
    size: int
    activation: str
    dropout_rate: float = 0.0
    use_bias: bool = True

class Layer:
    def __init__(self, config: LayerConfig):
        self.size = config.size
        self.activation_func = ActivationFunctions.get_activation_pair(config.activation)
        self.dropout_rate = config.dropout_rate
        self.use_bias = config.use_bias
        self.weights: Optional[Matrix] = None
        self.bias: Optional[Matrix] = None
        self.last_input: Optional[Matrix] = None
        self.last_output: Optional[Matrix] = None
        self.dropout_mask: Optional[Matrix] = None

class EnhancedNeuralNetwork:
    def __init__(self, architecture: List[LayerConfig], num_threads: int = 4):
        self.layers = [Layer(config) for config in architecture]
        self.num_threads = num_threads
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights using He initialization"""
        for i in range(len(self.layers) - 1):
            current_layer = self.layers[i]
            next_layer = self.layers[i + 1]
            
            # He initialization scale
            scale = np.sqrt(2.0 / current_layer.size)
            
            # Initialize weights
            weights = Matrix(current_layer.size, next_layer.size)
            weights.data = np.random.randn(current_layer.size, next_layer.size) * scale
            current_layer.weights = weights
            
            # Initialize biases if used
            if next_layer.use_bias:
                bias = Matrix(1, next_layer.size)
                bias.data = np.zeros((1, next_layer.size))
                current_layer.bias = bias

    def forward(self, X: Matrix, training: bool = False) -> Matrix:
        """Forward propagation with dropout"""
        current_input = X
        
        for i, layer in enumerate(self.layers[:-1]):
            # Store input for backprop
            layer.last_input = current_input
            
            # Compute layer output
            output = current_input.multiply(layer.weights, self.num_threads)
            
            if layer.bias is not None:
                # Broadcast bias
                output = output.add(Matrix.from_numpy(
                    np.broadcast_to(layer.bias.data, (output.rows, layer.bias.cols))
                ))
            
            # Apply activation
            output = output.map(layer.activation_func.forward)
            
            # Apply dropout during training
            if training and layer.dropout_rate > 0:
                mask = Matrix(output.rows, output.cols)
                scale = 1.0 / (1.0 - layer.dropout_rate)
                mask.data = np.where(
                    np.random.random(output.data.shape) > layer.dropout_rate,
                    scale, 0
                )
                layer.dropout_mask = mask
                output = output.multiply_elementwise(mask)
            
            layer.last_output = output
            current_input = output
        
        return current_input

    def backward(self, X: Matrix, y: Matrix, learning_rate: float) -> float:
        """Backward propagation with gradient clipping"""
        batch_size = X.rows
        output = self.layers[-2].last_output
        error = output.subtract(y)
        
        # Calculate initial loss
        loss = error.multiply_elementwise(error).sum() / (2 * batch_size)
        
        # Gradient clipping
        error = error.clip(-1.0, 1.0)
        
        for i in range(len(self.layers) - 2, -1, -1):
            layer = self.layers[i]
            
            # Apply dropout mask if used
            if layer.dropout_mask is not None:
                error = error.multiply_elementwise(layer.dropout_mask)
            
            # Calculate gradients
            input_transpose = layer.last_input.transpose()
            weight_gradients = input_transpose.multiply(error, self.num_threads)
            
            # Calculate bias gradients
            if layer.bias is not None:
                bias_gradients = error.sum(axis=0)
                bias_gradients.data /= batch_size
            
            # Update weights and biases
            weight_gradients.data /= batch_size
            layer.weights = layer.weights.subtract(
                Matrix.from_numpy(weight_gradients.data * learning_rate)
            )
            
            if layer.bias is not None:
                layer.bias = layer.bias.subtract(
                    Matrix.from_numpy(bias_gradients.data * learning_rate)
                )
            
            # Calculate error for next layer
            if i > 0:
                weights_transpose = layer.weights.transpose()
                error = error.multiply(weights_transpose, self.num_threads)
                error = error.multiply_elementwise(
                    layer.last_input.map(layer.activation_func.backward)
                )
        
        return float(loss)

    def train(self, X: Matrix, y: Matrix, epochs: int = 1000, 
              batch_size: int = 32, learning_rate: float = 0.1) -> Dict:
        """Train using mini-batch gradient descent with progress reporting"""
        history = {'loss': [], 'accuracy': [], 'time_per_epoch': []}
        n_samples = X.rows
        
        print(f"\nTraining with {epochs} epochs, batch size {batch_size}")
        for epoch in tqdm(range(epochs), desc="Training"):
            epoch_start = time.time()
            
            # Shuffle indices
            indices = np.random.permutation(n_samples)
            epoch_loss = 0.0
            
            # Mini-batch training
            for start_idx in range(0, n_samples, batch_size):
                batch_indices = indices[start_idx:start_idx + batch_size]
                
                # Create batch matrices
                X_batch = Matrix.from_numpy(X.data[batch_indices])
                y_batch = Matrix.from_numpy(y.data[batch_indices])
                
                # Forward and backward pass
                self.forward(X_batch, training=True)
                batch_loss = self.backward(X_batch, y_batch, learning_rate)
                epoch_loss += batch_loss * len(batch_indices)
            
            # Calculate metrics
            epoch_loss /= n_samples
            predictions = self.forward(X, training=False)
            
            correct = 0
            for i in range(X.rows):
                predicted = np.argmax(predictions.data[i])
                actual = np.argmax(y.data[i])
                if predicted == actual:
                    correct += 1
            accuracy = correct / X.rows
            
            # Store metrics
            epoch_time = time.time() - epoch_start
            history['loss'].append(epoch_loss)
            history['accuracy'].append(accuracy)
            history['time_per_epoch'].append(epoch_time)
            
            if epoch % 10 == 0:
                tqdm.write(
                    f"Epoch {epoch}: Loss={epoch_loss:.4f}, "
                    f"Accuracy={accuracy:.4f}, Time={epoch_time:.2f}s"
                )
        
        return history
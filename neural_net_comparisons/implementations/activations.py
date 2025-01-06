import math
from dataclasses import dataclass
from typing import Callable, List, Dict


@dataclass
class ActivationFunction:
    forward: Callable[[float], float]
    backward: Callable[[float], float]
    name: str


class ActivationFunctions:
    @staticmethod
    def sigmoid(x: float) -> float:
        if x > 32:  # Avoid overflow
            return 1.0
        if x < -32:  # Avoid underflow
            return 0.0
        return 1.0 / (1.0 + math.exp(-x))

    @staticmethod
    def sigmoid_derivative(x: float) -> float:
        sx = ActivationFunctions.sigmoid(x)
        return sx * (1.0 - sx)

    @staticmethod
    def relu(x: float) -> float:
        return max(0.0, x)

    @staticmethod
    def relu_derivative(x: float) -> float:
        return 1.0 if x > 0 else 0.0

    @staticmethod
    def tanh(x: float) -> float:
        return math.tanh(x)

    @staticmethod
    def tanh_derivative(x: float) -> float:
        tx = math.tanh(x)
        return 1.0 - tx * tx

    @staticmethod
    def leaky_relu(x: float, alpha: float = 0.01) -> float:
        return x if x > 0 else alpha * x

    @staticmethod
    def leaky_relu_derivative(x: float, alpha: float = 0.01) -> float:
        return 1.0 if x > 0 else alpha

    @staticmethod
    def get_activation_pair(name: str) -> ActivationFunction:
        """Get activation function and its derivative by name"""
        activations: Dict[str, ActivationFunction] = {
            "sigmoid": ActivationFunction(
                ActivationFunctions.sigmoid,
                ActivationFunctions.sigmoid_derivative,
                "sigmoid",
            ),
            "relu": ActivationFunction(
                ActivationFunctions.relu,
                ActivationFunctions.relu_derivative,
                "relu"
            ),
            "tanh": ActivationFunction(
                ActivationFunctions.tanh, ActivationFunctions.tanh_derivative,
                "tanh"
            ),
            "leaky_relu": ActivationFunction(
                lambda x: ActivationFunctions.leaky_relu(x),
                lambda x: ActivationFunctions.leaky_relu_derivative(x),
                "leaky_relu",
            ),
        }

        if name not in activations:
            raise ValueError(f"Unknown activation function: {name}")

        return activations[name]

    @staticmethod
    def softmax(x: list[float]) -> list[float]:
        """Softmax activation for output layer"""
        # Subtract max for numerical stability
        exp_x = [math.exp(i - max(x)) for i in x]
        sum_exp = sum(exp_x)
        return [i / sum_exp for i in exp_x]


class Activations:
    """Collection of activation functions with their derivatives"""

    @staticmethod
    def sigmoid(x: float) -> float:
        """Sigmoid activation function"""
        return 1.0 / (1.0 + math.exp(-x))

    @staticmethod
    def sigmoid_derivative(x: float) -> float:
        """Derivative of sigmoid function"""
        sx = Activations.sigmoid(x)
        return sx * (1 - sx)

    @staticmethod
    def tanh(x: float) -> float:
        """Hyperbolic tangent activation function"""
        return math.tanh(x)

    @staticmethod
    def tanh_derivative(x: float) -> float:
        """Derivative of hyperbolic tangent function"""
        return 1 - math.tanh(x) ** 2

    @staticmethod
    def relu(x: float) -> float:
        """Rectified Linear Unit activation function"""
        return max(0.0, x)

    @staticmethod
    def relu_derivative(x: float) -> float:
        """Derivative of ReLU function"""
        return 1.0 if x > 0 else 0.0

    @staticmethod
    def leaky_relu(x: float, alpha: float = 0.01) -> float:
        """Leaky ReLU activation function"""
        return x if x > 0 else alpha * x

    @staticmethod
    def leaky_relu_derivative(x: float, alpha: float = 0.01) -> float:
        """Derivative of Leaky ReLU function"""
        return 1.0 if x > 0 else alpha

    @staticmethod
    def softmax(x: List[float]) -> List[float]:
        """Softmax activation function"""
        # Subtract max for numerical stability
        exp_x = [math.exp(i - max(x)) for i in x]
        sum_exp = sum(exp_x)
        return [i / sum_exp for i in exp_x]

    @staticmethod
    def softmax_derivative(x: List[float]) -> List[List[float]]:
        """Jacobian matrix of softmax function"""
        s = Activations.softmax(x)
        jacobian = []
        for i in range(len(s)):
            row = []
            for j in range(len(s)):
                if i == j:
                    row.append(s[i] * (1 - s[i]))
                else:
                    row.append(-s[i] * s[j])
            jacobian.append(row)
        return jacobian


class ActivationFunctionFactory:
    """Factory for creating activation functions"""

    @staticmethod
    def get(name: str) -> ActivationFunction:
        """Get activation function by name"""
        functions = {
            "sigmoid": ActivationFunction(
                Activations.sigmoid, Activations.sigmoid_derivative, "sigmoid"
            ),
            "tanh": ActivationFunction(
                Activations.tanh, Activations.tanh_derivative, "tanh"
            ),
            "relu": ActivationFunction(
                Activations.relu, Activations.relu_derivative, "relu"
            ),
            "leaky_relu": ActivationFunction(
                Activations.leaky_relu,
                Activations.leaky_relu_derivative,
                "leaky_relu"
            ),
        }

        if name not in functions:
            raise ValueError(f"Unknown activation function: {name}")

        return functions[name]

import math
import random
from typing import Tuple, List
from ..implementations.matrix import Matrix
from ..implementations.python_impl import LayerConfig


class StressTestGenerator:
    @staticmethod
    def generate_mnist_like(
        num_samples: int, input_size: int = 784
    ) -> Tuple[Matrix, Matrix]:
        """Generate MNIST-like dataset with complex patterns"""
        X = Matrix(num_samples, input_size)
        y = Matrix(num_samples, 10)  # 10 classes

        width = int(math.sqrt(input_size))  # Assuming square image

        for i in range(num_samples):
            digit = i % 10
            pattern = []

            for row in range(width):
                for col in range(width):
                    pixel = 0.0
                    normalized_row = row / width
                    normalized_col = col / width

                    if digit == 0:  # Circle
                        center_dist = math.sqrt((normalized_row - 0.5)**2 + (normalized_col - 0.5)**2)
                        pixel = 1.0 if 0.3 < center_dist < 0.4 else 0.0

                    elif digit == 1:  # Vertical line
                        pixel = 1.0 if 0.45 <= normalized_col <= 0.55 else 0.0

                    elif digit == 2:  # Sine wave
                        pixel = math.sin(normalized_row * 6) * 0.5 + 0.5
                        pixel = 1.0 if 0.3 <= normalized_col - pixel * 0.3 <= 0.4 else 0.0

                    elif digit == 3:  # Spiral
                        angle = math.atan2(normalized_row - 0.5, normalized_col - 0.5)
                        radius = math.sqrt((normalized_row - 0.5)**2 + (normalized_col - 0.5)**2)
                        pixel = 1.0 if abs(radius - angle / (2 * math.pi)) < 0.1 else 0.0

                    else:  # Complex patterns for other digits
                        freq = 1 + digit * 0.5
                        pixel = math.sin(normalized_row * freq * math.pi) * math.cos(normalized_col * freq * math.pi)
                        pixel = max(0, min(1, (pixel + 1) / 2))

                    # Add some noise
                    pixel += random.gauss(0, 0.1)
                    pixel = max(0, min(1, pixel))
                    pattern.append(pixel)

            X.data[i] = pattern
            y.data[i] = [1.0 if j == digit else 0.0 for j in range(10)]

        return X, y

    @staticmethod
    def generate_stress_test(
        config: str = 'medium'
    ) -> Tuple[Matrix, Matrix, List[LayerConfig]]:
        """Generate complete stress test data and architecture"""
        configs = {
            'small': (1000, 784),
            'medium': (5000, 784),
            'large': (10000, 784),
            'xlarge': (20000, 1024)
        }

        architectures = {
            'small': [
                LayerConfig(784, 'relu'),
                LayerConfig(128, 'relu', dropout_rate=0.2),
                LayerConfig(64, 'relu', dropout_rate=0.2),
                LayerConfig(10, 'sigmoid')
            ],
            'medium': [
                LayerConfig(784, 'relu'),
                LayerConfig(256, 'relu', dropout_rate=0.2),
                LayerConfig(128, 'relu', dropout_rate=0.2),
                LayerConfig(64, 'relu', dropout_rate=0.2),
                LayerConfig(10, 'sigmoid')
            ],
            'large': [
                LayerConfig(784, 'relu'),
                LayerConfig(512, 'relu', dropout_rate=0.3),
                LayerConfig(256, 'relu', dropout_rate=0.3),
                LayerConfig(128, 'relu', dropout_rate=0.2),
                LayerConfig(64, 'relu', dropout_rate=0.2),
                LayerConfig(10, 'sigmoid')
            ],
            'xlarge': [
                LayerConfig(1024, 'relu'),
                LayerConfig(1024, 'relu', dropout_rate=0.4),
                LayerConfig(512, 'relu', dropout_rate=0.3),
                LayerConfig(256, 'relu', dropout_rate=0.3),
                LayerConfig(128, 'relu', dropout_rate=0.2),
                LayerConfig(10, 'sigmoid')
            ]
        }

        num_samples, input_size = configs.get(config, configs['medium'])
        architecture = architectures.get(config, architectures['medium'])

        X, y = StressTestGenerator.generate_mnist_like(num_samples, input_size)

        return X, y, architecture

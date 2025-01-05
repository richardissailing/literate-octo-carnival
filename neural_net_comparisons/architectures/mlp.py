from typing import List
from ..implementations.python_impl import LayerConfig

class MLPArchitectures:
    @staticmethod
    def create_classifier(input_size: int, 
                         hidden_sizes: List[int], 
                         output_size: int,
                         hidden_activation: str = 'relu',
                         output_activation: str = 'sigmoid',
                         dropout_rate: float = 0.0) -> List[LayerConfig]:
        """Create a multi-layer perceptron classifier architecture"""
        layers = []
        
        # Input layer
        layers.append(LayerConfig(input_size, hidden_activation))
        
        # Hidden layers
        for size in hidden_sizes:
            layers.append(LayerConfig(
                size=size,
                activation=hidden_activation,
                dropout_rate=dropout_rate
            ))
        
        # Output layer
        layers.append(LayerConfig(
            size=output_size,
            activation=output_activation,
            dropout_rate=0.0  # No dropout in output layer
        ))
        
        return layers
    
    @staticmethod
    def create_regressor(input_size: int,
                        hidden_sizes: List[int],
                        output_size: int = 1,
                        hidden_activation: str = 'relu',
                        output_activation: str = 'linear') -> List[LayerConfig]:
        """Create a multi-layer perceptron regressor architecture"""
        return MLPArchitectures.create_classifier(
            input_size=input_size,
            hidden_sizes=hidden_sizes,
            output_size=output_size,
            hidden_activation=hidden_activation,
            output_activation=output_activation,
            dropout_rate=0.0  # Typically less dropout in regression
        )
    
    @staticmethod
    def create_wide_and_deep(input_size: int,
                            wide_size: int,
                            deep_layers: List[int],
                            output_size: int) -> List[LayerConfig]:
        """Create a wide and deep network architecture"""
        layers = []
        
        # Input layer
        layers.append(LayerConfig(input_size, 'linear'))
        
        # Wide path (direct connection)
        layers.append(LayerConfig(wide_size, 'linear'))
        
        # Deep path
        for size in deep_layers:
            layers.append(LayerConfig(
                size=size,
                activation='relu',
                dropout_rate=0.2
            ))
        
        # Combine wide and deep paths
        combined_size = wide_size + deep_layers[-1]
        layers.append(LayerConfig(combined_size, 'relu'))
        
        # Output layer
        layers.append(LayerConfig(output_size, 'sigmoid'))
        
        return layers
    
    @staticmethod
    def create_residual_mlp(input_size: int,
                           hidden_size: int,
                           num_residual_blocks: int,
                           output_size: int) -> List[LayerConfig]:
        """Create a multi-layer perceptron with residual connections"""
        layers = []
        
        # Input layer
        layers.append(LayerConfig(input_size, 'relu'))
        
        # Initial projection to hidden size if needed
        if input_size != hidden_size:
            layers.append(LayerConfig(hidden_size, 'relu'))
        
        # Residual blocks
        for _ in range(num_residual_blocks):
            layers.extend([
                LayerConfig(hidden_size, 'relu'),
                LayerConfig(hidden_size, 'relu')
            ])
        
        # Output layer
        layers.append(LayerConfig(output_size, 'sigmoid'))
        
        return layers
    
    @staticmethod
    def create_highway_network(input_size: int,
                             hidden_sizes: List[int],
                             output_size: int) -> List[LayerConfig]:
        """Create a highway network architecture"""
        layers = []
        
        # Input layer
        layers.append(LayerConfig(input_size, 'relu'))
        
        # Highway layers
        for size in hidden_sizes:
            # Transform gate
            layers.append(LayerConfig(size, 'sigmoid'))
            # Main layer
            layers.append(LayerConfig(size, 'relu'))
            # Carry gate (implicitly 1 - transform gate)
        
        # Output layer
        layers.append(LayerConfig(output_size, 'sigmoid'))
        
        return layers
from typing import List, Optional
from ..implementations.python_impl import LayerConfig

class ResidualArchitectures:
    @staticmethod
    def create_resnet_block(input_size: int,
                           hidden_size: Optional[int] = None,
                           activation: str = 'relu') -> List[LayerConfig]:
        """Create a single residual block"""
        if hidden_size is None:
            hidden_size = input_size
            
        layers = []
        
        # First layer
        layers.append(LayerConfig(hidden_size, activation))
        
        # Second layer (output size must match input for residual connection)
        layers.append(LayerConfig(input_size, 'linear'))  # Linear activation before residual addition
        
        return layers
    
    @staticmethod
    def create_residual_network(input_size: int,
                              hidden_size: int,
                              num_blocks: int,
                              output_size: int) -> List[LayerConfig]:
        """Create a complete residual network"""
        layers = []
        
        # Initial layer
        layers.append(LayerConfig(input_size, 'relu'))
        
        # If input_size != hidden_size, add projection layer
        if input_size != hidden_size:
            layers.append(LayerConfig(hidden_size, 'relu'))
        
        # Add residual blocks
        for _ in range(num_blocks):
            layers.extend(ResidualArchitectures.create_resnet_block(hidden_size))
        
        # Final output layer
        layers.append(LayerConfig(output_size, 'sigmoid'))
        
        return layers
    
    @staticmethod
    def create_dense_residual_network(input_size: int,
                                    growth_rate: int,
                                    num_blocks: int,
                                    output_size: int) -> List[LayerConfig]:
        """Create a densely connected residual network"""
        layers = []
        current_size = input_size
        
        # Initial layer
        layers.append(LayerConfig(current_size, 'relu'))
        
        # Dense blocks
        for _ in range(num_blocks):
            # Add layers with growing size
            layers.append(LayerConfig(growth_rate, 'relu'))
            current_size += growth_rate
            
            # Add transition layer to control size
            transition_size = current_size // 2
            layers.append(LayerConfig(transition_size, 'relu'))
            current_size = transition_size
        
        # Final output layer
        layers.append(LayerConfig(output_size, 'sigmoid'))
        
        return layers
    
    @staticmethod
    def create_pyramid_residual_network(input_size: int,
                                      initial_channels: int,
                                      num_stages: int,
                                      blocks_per_stage: int,
                                      output_size: int) -> List[LayerConfig]:
        """Create a pyramid-shaped residual network"""
        layers = []
        current_channels = initial_channels
        
        # Initial convolution
        layers.append(LayerConfig(current_channels, 'relu'))
        
        # Create stages
        for stage in range(num_stages):
            # Double channels at each stage
            if stage > 0:
                current_channels *= 2
            
            # Add blocks for this stage
            for block in range(blocks_per_stage):
                layers.extend(ResidualArchitectures.create_resnet_block(
                    current_channels,
                    current_channels
                ))
        
        # Global average pooling (implicit in forward pass)
        
        # Final output layer
        layers.append(LayerConfig(output_size, 'sigmoid'))
        
        return layers
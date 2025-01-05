from typing import List
from ..implementations.python_impl import LayerConfig

class AutoencoderArchitectures:
    @staticmethod
    def create_basic_autoencoder(input_size: int,
                                encoding_sizes: List[int],
                                activation: str = 'relu',
                                output_activation: str = 'sigmoid') -> List[LayerConfig]:
        """Create a basic autoencoder architecture"""
        layers = []
        
        # Input layer
        layers.append(LayerConfig(input_size, activation))
        
        # Encoder layers
        for size in encoding_sizes:
            layers.append(LayerConfig(size, activation))
        
        # Decoder layers (reverse of encoder)
        for size in reversed(encoding_sizes[:-1]):
            layers.append(LayerConfig(size, activation))
        
        # Output layer (reconstruction)
        layers.append(LayerConfig(input_size, output_activation))
        
        return layers
    
    @staticmethod
    def create_variational_autoencoder(input_size: int,
                                     encoding_sizes: List[int],
                                     latent_dim: int) -> List[LayerConfig]:
        """Create a variational autoencoder architecture"""
        layers = []
        
        # Encoder
        layers.append(LayerConfig(input_size, 'relu'))
        for size in encoding_sizes:
            layers.append(LayerConfig(size, 'relu'))
        
        # Latent space (mean and log variance)
        layers.append(LayerConfig(latent_dim * 2, 'linear'))
        
        # Sampling layer (handled in forward pass)
        layers.append(LayerConfig(latent_dim, 'linear'))
        
        # Decoder
        for size in reversed(encoding_sizes):
            layers.append(LayerConfig(size, 'relu'))
        
        # Reconstruction layer
        layers.append(LayerConfig(input_size, 'sigmoid'))
        
        return layers
    
    @staticmethod
    def create_denoising_autoencoder(input_size: int,
                                   encoding_sizes: List[int],
                                   noise_level: float = 0.1) -> List[LayerConfig]:
        """Create a denoising autoencoder architecture"""
        layers = []
        
        # Input layer with noise
        layers.append(LayerConfig(
            size=input_size,
            activation='relu',
            dropout_rate=noise_level  # Use dropout as noise
        ))
        
        # Encoder
        for size in encoding_sizes:
            layers.append(LayerConfig(size, 'relu'))
        
        # Decoder
        for size in reversed(encoding_sizes[:-1]):
            layers.append(LayerConfig(size, 'relu'))
        
        # Clean reconstruction
        layers.append(LayerConfig(input_size, 'sigmoid'))
        
        return layers
    
    @staticmethod
    def create_sparse_autoencoder(input_size: int,
                                encoding_sizes: List[int],
                                sparsity_level: float = 0.05) -> List[LayerConfig]:
        """Create a sparse autoencoder architecture"""
        layers = []
        
        # Input layer
        layers.append(LayerConfig(input_size, 'relu'))
        
        # Encoder with sparsity constraint
        for size in encoding_sizes:
            layers.append(LayerConfig(
                size=size,
                activation='relu',
                dropout_rate=1.0 - sparsity_level  # Enforce sparsity through dropout
            ))
        
        # Decoder
        for size in reversed(encoding_sizes[:-1]):
            layers.append(LayerConfig(size, 'relu'))
        
        # Reconstruction layer
        layers.append(LayerConfig(input_size, 'sigmoid'))
        
        return layers
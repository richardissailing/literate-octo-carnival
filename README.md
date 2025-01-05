# Neural Network Implementation Comparison

A comprehensive toolkit for comparing neural network implementations across Python, Go, and TypeScript, focusing on pure implementations without external ML libraries. This project provides insights into performance characteristics, implementation differences, and optimization techniques across different programming languages.

## Features

### Core Functionality
- Pure implementations in Python, Go, and TypeScript
- Matrix operations built from scratch
- Support for parallel processing
- Comprehensive benchmarking suite
- Performance profiling tools

### Neural Network Features
- Feed-forward neural networks
- Mini-batch gradient descent
- Configurable architectures
- Multiple activation functions
- Dropout regularization
- Parallel processing support

### Benchmarking Capabilities
- Performance profiling
- Memory usage tracking
- CPU utilization monitoring
- Training metrics comparison
- Cross-language analysis

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/neural-net-comparisons
cd neural-net-comparisons

# Install with Poetry
poetry install

# Install language requirements
npm install -g typescript  # For TypeScript implementation
```

## Usage

### Basic Benchmark
```bash
# Run default XOR problem benchmark
poetry run benchmark --verbose
```

### Performance Testing
```bash
# Run medium-size test
poetry run benchmark --test-size medium --epochs 50 --batch-size 256 --num-threads 8

# Run large-size test
poetry run benchmark --test-size large --epochs 25 --batch-size 512 --num-threads 12
```

### Available Test Configurations
- `xor`: Basic XOR problem (4 samples)
- `small`: 1,000 samples, 784 dimensions
- `medium`: 5,000 samples, 784 dimensions
- `large`: 10,000 samples, 784 dimensions
- `xlarge`: 20,000 samples, 1024 dimensions

### Command Line Options
```bash
poetry run benchmark --help

Options:
  --test-size {xor,small,medium,large,xlarge}
  --epochs EPOCHS                 Number of training epochs
  --batch-size BATCH_SIZE        Training batch size
  --learning-rate LEARNING_RATE  Learning rate for training
  --num-threads NUM_THREADS      Number of threads for parallel processing
  --implementations {python,go,typescript}
  --verbose                      Enable detailed output
  --output OUTPUT               Output file for results
```

## Project Structure

```
neural_net_comparisons/
├── neural_net_comparisons/
│   ├── implementations/
│   │   ├── matrix.py           # Optimized matrix operations
│   │   ├── python_impl.py      # Python neural network
│   │   ├── go_impl.go         # Go neural network
│   │   └── typescript_impl.ts  # TypeScript neural network
│   ├── benchmark/
│   │   ├── runner.py          # Benchmark orchestration
│   │   └── profiler.py        # Performance profiling
│   └── test_generators/
│       └── complex_test.py    # Test data generation
```

## Implementation Details

### Matrix Operations
- Custom matrix implementation optimized for performance
- Support for parallel processing with ThreadPoolExecutor
- Automatic optimization based on matrix size
- Efficient memory usage patterns

### Neural Network Features
- Configurable layer architectures
- Multiple activation functions (ReLU, Sigmoid, Tanh)
- Dropout regularization
- Mini-batch gradient descent
- Parallel processing capabilities

### Performance Optimizations
- Multi-threaded matrix operations
- Batch size auto-adjustment
- Memory-efficient data structures
- Smart thread allocation
- Small matrix optimization path

## Benchmark Output

```json
{
  "Python": {
    "training_time": 2.776,
    "accuracy": 95.2,
    "final_loss": 0.089273
  },
  "Go": {
    "training_time": 1.234,
    "accuracy": 94.8,
    "final_loss": 0.092847
  },
  "TypeScript": {
    "training_time": 3.897,
    "accuracy": 93.5,
    "final_loss": 0.095632
  }
}
```

## Profiling Metrics

- Execution time
- Memory usage
- CPU utilization
- Training convergence
- Resource efficiency

## Contributing

1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Add/update tests
5. Submit a pull request

## Future Improvements

- GPU support
- Additional architectures
- More activation functions
- Distributed training
- Web visualization interface


---

Note: This project is designed for educational purposes and performance analysis. For production machine learning applications, consider using established frameworks like TensorFlow or PyTorch.
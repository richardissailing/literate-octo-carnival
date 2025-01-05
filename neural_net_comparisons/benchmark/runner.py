import json
import subprocess
from typing import List, Optional, Tuple, Dict
from pathlib import Path
import argparse
from datetime import datetime
from .profiler import Profiler
from typing import List, Optional, Tuple, Dict
from ..test_generators.complex_test import StressTestGenerator


class BenchmarkRunner:
    def __init__(self, args: argparse.Namespace):
        self.epochs = args.epochs
        self.batch_size = args.batch_size
        self.learning_rate = args.learning_rate
        self.output_file = args.output
        self.implementations = args.implementations
        self.verbose = args.verbose
        self.test_size = getattr(args, 'test_size', 'xor')
        self.num_threads = getattr(args, 'num_threads', 4)  # Default to 4 threads
        self.results = {}
        self.all_losses = {}
        self.profiling_results = {}
        
        # Adjust batch size based on test size
        if self.test_size == 'large':
            self.batch_size = max(256, self.batch_size)
        elif self.test_size == 'xlarge':
            self.batch_size = max(512, self.batch_size)
        
        # Generate test data based on configuration
        if self.test_size != 'xor':
            print(f"\nGenerated {self.test_size} test data:")
            self.X, self.y, self.architecture = StressTestGenerator.generate_stress_test(self.test_size)
            print(f"Samples: {self.X.rows}")
            print(f"Input dimensions: {self.X.cols}")
            print(f"Architecture: {[layer.size for layer in self.architecture]}")
            print(f"Hidden layers: {[layer.size for layer in self.architecture[1:-1]]}\n")
        else:
            self.X = None
            self.y = None
            self.architecture = None

    def serialize_profiling_results(self):
        """Convert profiling results to JSON-serializable format"""
        serializable_results = {}
        for lang, stats in self.profiling_results.items():
            # Check if stats is already a dict or a ProfileStats object
            if isinstance(stats, dict):
                serializable_results[lang] = {
                    'execution_time': stats.get('execution_time', 0),
                    'memory_peak': stats.get('memory_peak', 0),
                    'memory_diff': stats.get('memory_diff', 0),
                    'cpu_percent': stats.get('cpu_percent', 0)
                }
            else:
                # If it's a ProfileStats object
                serializable_results[lang] = {
                    'execution_time': getattr(stats, 'execution_time', 0),
                    'memory_peak': getattr(stats, 'memory_peak', 0),
                    'memory_diff': getattr(stats, 'memory_diff', 0),
                    'cpu_percent': getattr(stats, 'cpu_percent', 0)
                }
                
            # Filter out any Nones or non-JSON-serializable values
            serializable_results[lang] = {
                k: v if isinstance(v, (int, float, str, bool)) else str(v)
                for k, v in serializable_results[lang].items()
                if v is not None
            }
        
        return serializable_results

    def print_analysis(self):
        """Print analysis of the benchmark results"""
        if not self.results:
            print("\nNo results to analyze")
            return

        print("\nBenchmark Analysis:")
        print("=" * 50)
        
        # Performance comparison
        times = {k: v['training_time'] for k, v in self.results.items() if v}
        if times:
            fastest = min(times.items(), key=lambda x: x[1])
            print(f"\nPerformance Rankings:")
            for lang, time in sorted(times.items(), key=lambda x: x[1]):
                speedup = time / fastest[1]
                print(f"{lang:12} {time:.3f}s ({speedup:.1f}x slower than {fastest[0]})")
        
        # Accuracy comparison
        accuracies = {k: v['accuracy'] for k, v in self.results.items() if v}
        if accuracies:
            print(f"\nAccuracy Rankings:")
            for lang, acc in sorted(accuracies.items(), key=lambda x: x[1], reverse=True):
                print(f"{lang:12} {acc:.1f}%")
        
        # Loss comparison
        losses = {k: v['final_loss'] for k, v in self.results.items() if v}
        if losses:
            print(f"\nFinal Loss Rankings:")
            for lang, loss in sorted(losses.items(), key=lambda x: x[1]):
                print(f"{lang:12} {loss:.6f}")
        
        # Profile summary
        if self.profiling_results:
            print("\nResource Usage Summary:")
            profile_summary = self.serialize_profiling_results()
            for lang, stats in profile_summary.items():
                print(f"\n{lang}:")
                print(f"  Memory Peak:   {stats.get('memory_peak', 0):.1f} MB")
                print(f"  Memory Growth: {stats.get('memory_diff', 0):.1f} MB")
                print(f"  CPU Usage:     {stats.get('cpu_percent', 0):.1f}%")


    def log(self, message: str):
        """Conditional logging based on verbose flag"""
        if self.verbose:
            print(message)

    def get_test_data(self):
        """Get test data - either stress test or default XOR"""
        if self.X is not None and self.y is not None and self.architecture is not None:
            return self.X, self.y, self.architecture
        
        # Default XOR problem
        from ..implementations.matrix import Matrix
        from ..implementations.python_impl import LayerConfig
        
        X = Matrix(4, 2)
        X.data = [[0, 0], [0, 1], [1, 0], [1, 1]]
        
        y = Matrix(4, 1)
        y.data = [[0], [1], [1], [0]]
        
        architecture = [
            LayerConfig(2, 'sigmoid'),
            LayerConfig(4, 'sigmoid'),
            LayerConfig(1, 'sigmoid')
        ]
        
        return X, y, architecture

    def run_python_benchmark(self) -> Tuple[Optional[float], Optional[float], Optional[List[float]], Optional[Dict]]:
        """Run Python neural network implementation"""
        try:
            self.log("\nRunning Python implementation...")
            
            X, y, architecture = self.get_test_data()
            
            profiler = Profiler(enabled=True, name="Python")
            with profiler:
                from ..implementations.python_impl import EnhancedNeuralNetwork
                
                # Initialize network with architecture
                nn = EnhancedNeuralNetwork(architecture)
                
                # Train
                history = nn.train(
                    X=X,
                    y=y,
                    epochs=self.epochs,
                    batch_size=self.batch_size,
                    learning_rate=self.learning_rate
                )
            
            if self.verbose:
                profiler.print_stats()
            
            profiler.save_stats(Path('profiling_results'), 'python_impl')
            
            return (profiler.stats.execution_time if profiler.stats else None,
                    history['accuracy'][-1],
                    history['loss'],
                    profiler.stats.__dict__ if profiler.stats else None)
        except Exception as e:
            self.log(f"Error in Python benchmark: {e}")
            if self.verbose:
                import traceback
                traceback.print_exc()
            return None, None, None, None

    def run_go_benchmark(self) -> Tuple[Optional[float], Optional[float], Optional[List[float]], Optional[Dict]]:
        """Run Go neural network implementation"""
        try:
            self.log("\nRunning Go implementation...")
            
            profiler = Profiler(enabled=True, name="Go")
            with profiler:
                go_path = Path(__file__).parent.parent / 'implementations' / 'go_impl.go'
                
                # Run Go program with parameters
                cmd = [
                    'go', 'run', str(go_path),
                    '--epochs', str(self.epochs),
                    '--batch-size', str(self.batch_size),
                    '--learning-rate', str(self.learning_rate)
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if self.verbose:
                    print("Go Output:", result.stdout)
                    print("Go Errors:", result.stderr)
                
                # Parse output
                losses = []
                accuracy = None
                
                for line in result.stdout.split('\n'):
                    if line.startswith('Epoch'):
                        try:
                            loss = float(line.split('Loss: ')[1])
                            losses.append(loss)
                        except (IndexError, ValueError) as e:
                            self.log(f"Error parsing loss: {e}")
                    elif line.startswith('Final Accuracy:'):
                        try:
                            accuracy = float(line.split(': ')[1])
                        except (IndexError, ValueError) as e:
                            self.log(f"Error parsing accuracy: {e}")
            
            if self.verbose:
                profiler.print_stats()
            
            profiler.save_stats(Path('profiling_results'), 'go_impl')
            
            return (profiler.stats.execution_time if profiler.stats else None,
                    accuracy,
                    losses,
                    profiler.stats.__dict__ if profiler.stats else None)
        except Exception as e:
            self.log(f"Error in Go benchmark: {e}")
            if self.verbose:
                import traceback
                traceback.print_exc()
            return None, None, None, None

    def run_typescript_benchmark(self) -> Tuple[Optional[float], Optional[float], Optional[List[float]], Optional[Dict]]:
        """Run TypeScript neural network implementation"""
        try:
            self.log("\nRunning TypeScript implementation...")
            
            profiler = Profiler(enabled=True, name="Typescript")
            with profiler:
                # Paths
                ts_path = Path(__file__).parent.parent / 'implementations' / 'typescript_impl.ts'
                js_path = ts_path.with_suffix('.js')
                
                # Compile TypeScript
                compile_result = subprocess.run(['tsc', str(ts_path)], capture_output=True, text=True)
                if compile_result.returncode != 0:
                    self.log(f"TypeScript compilation failed: {compile_result.stderr}")
                    return None, None, None, None
                
                # Run with Node.js
                cmd = [
                    'node', str(js_path),
                    '--epochs', str(self.epochs),
                    '--batch-size', str(self.batch_size),
                    '--learning-rate', str(self.learning_rate)
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if self.verbose:
                    print("TypeScript Output:", result.stdout)
                    print("TypeScript Errors:", result.stderr)
                
                # Parse output
                losses = []
                accuracy = None
                
                for line in result.stdout.split('\n'):
                    if line.startswith('Epoch'):
                        try:
                            loss = float(line.split('Loss: ')[1])
                            losses.append(loss)
                        except (IndexError, ValueError) as e:
                            self.log(f"Error parsing loss: {e}")
                    elif line.startswith('Final Accuracy:'):
                        try:
                            accuracy = float(line.split(': ')[1])
                        except (IndexError, ValueError) as e:
                            self.log(f"Error parsing accuracy: {e}")
            
            if self.verbose:
                profiler.print_stats()
            
            profiler.save_stats(Path('profiling_results'), 'typescript_impl')
            
            return (profiler.stats.execution_time if profiler.stats else None,
                    accuracy,
                    losses,
                    profiler.stats.__dict__ if profiler.stats else None)
        except Exception as e:
            self.log(f"Error in TypeScript benchmark: {e}")
            if self.verbose:
                import traceback
                traceback.print_exc()
            return None, None, None, None

    def run(self):
        """Run all benchmarks"""
        print("\nStarting benchmarks...")
        print(f"Parameters: epochs={self.epochs}, batch_size={self.batch_size}, learning_rate={self.learning_rate}")
        
        implementations = {
            'python': self.run_python_benchmark,
            'go': self.run_go_benchmark,
            'typescript': self.run_typescript_benchmark
        }
        
        # Create profiling results directory
        Path('profiling_results').mkdir(exist_ok=True)
        
        # Run selected implementations
        for lang in self.implementations:
            if lang in implementations:
                time_result, accuracy, losses, profile_stats = implementations[lang]()
                
                if time_result is not None:
                    self.results[lang.capitalize()] = {
                        'training_time': round(time_result, 3),
                        'accuracy': round(accuracy * 100, 2) if accuracy is not None else None,
                        'final_loss': round(losses[-1], 6) if losses else None
                    }
                    self.all_losses[lang.capitalize()] = losses
                    if profile_stats:
                        self.profiling_results[lang.capitalize()] = profile_stats
                else:
                    self.results[lang.capitalize()] = None
        
        # Print results
        print("\nBenchmark Results:")
        print(json.dumps(self.results, indent=2))
        
        if self.verbose and self.profiling_results:
            print("\nProfiling Results:")
            serialized_profile = self.serialize_profiling_results()
            print(json.dumps(serialized_profile, indent=2))
        
        # Generate plots
        self.plot_results()
        
        # Save results
        self.save_results()

    def save_results(self):
        """Save benchmark results to file"""
        if not self.output_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_file = f'benchmark_results_{timestamp}.json'
        
        output = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'parameters': {
                    'epochs': self.epochs,
                    'batch_size': self.batch_size,
                    'learning_rate': self.learning_rate
                }
            },
            'results': self.results,
            'profiling_results': self.serialize_profiling_results() if self.profiling_results else None
        }
        
        with open(self.output_file, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"\nResults saved to '{self.output_file}'")

    def plot_results(self):
        """Generate visualization plots"""
        # Check if we have any loss data
        if not any(self.all_losses.values()):
            self.log("No loss data available for plotting")
            return
        
        # Import matplotlib and handle potential import error
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            self.log("matplotlib is not installed. Cannot create plots.")
            return
            
        # Create plots directory
        plots_dir = Path('plots')
        plots_dir.mkdir(exist_ok=True)
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        def create_loss_plot():
            """Create the training loss comparison plot"""
            plt.figure(figsize=(12, 6))
            
            for lang, losses in self.all_losses.items():
                if losses:
                    epochs = range(0, len(losses) * 100, 100)
                    plt.plot(epochs, losses, label=lang, marker='o')
            
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.title('Training Loss Comparison')
            plt.legend()
            plt.grid(True)
            
            plot_path = plots_dir / f'loss_comparison_{timestamp}.png'
            plt.savefig(plot_path)
            plt.close()
            return plot_path
            
        def create_performance_plot():
            """Create the performance metrics comparison plot"""
            if not self.profiling_results:
                return None
                
            plt.figure(figsize=(10, 6))
            languages = list(self.profiling_results.keys())
            metrics = ['execution_time', 'memory_peak', 'cpu_percent']
            
            x = range(len(languages))
            width = 0.25
            
            for i, metric in enumerate(metrics):
                try:
                    values = [self.profiling_results[lang][metric] 
                             for lang in languages]
                    plt.bar([xi + i*width for xi in x], values, width, 
                           label=metric)
                except KeyError as e:
                    self.log(f"Missing metric {e} for performance plot")
                    continue
            
            plt.xlabel('Implementation')
            plt.ylabel('Value')
            plt.title('Performance Metrics Comparison')
            plt.xticks([xi + width for xi in x], languages)
            plt.legend()
            
            plot_path = plots_dir / f'performance_comparison_{timestamp}.png'
            plt.savefig(plot_path)
            plt.close()
            return plot_path
        
        # Create plots
        loss_plot_path = None
        perf_plot_path = None
        
        try:
            loss_plot_path = create_loss_plot()
        except Exception as err:
            self.log(f"Error creating loss plot: {err}")
            
        try:
            perf_plot_path = create_performance_plot()
        except Exception as err:
            self.log(f"Error creating performance plot: {err}")
            
        # Report results
        if loss_plot_path or perf_plot_path:
            print("\nPlots saved in 'plots' directory:")
            if loss_plot_path:
                print(f"- Loss comparison: {loss_plot_path}")
            if perf_plot_path:
                print(f"- Performance comparison: {perf_plot_path}")
        else:
            self.log("No plots were generated due to errors")


    def save_results(self):
        """Save benchmark results to file"""
        if not self.output_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_file = f'benchmark_results_{timestamp}.json'
        
        output = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'parameters': {
                    'epochs': self.epochs,
                    'batch_size': self.batch_size,
                    'learning_rate': self.learning_rate
                }
            },
            'results': self.results,
            'profiling_results': self.serialize_profiling_results() if self.profiling_results else None
        }
        
        with open(self.output_file, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"\nResults saved to '{self.output_file}'")

def main():
    parser = argparse.ArgumentParser(description='Neural Network Implementation Benchmark')
    
    parser.add_argument('--epochs', type=int, default=50,
                      help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=128,
                      help='Training batch size')
    parser.add_argument('--learning-rate', type=float, default=0.1,
                      help='Learning rate for training')
    parser.add_argument('--output', type=str,
                      help='Output file for results (JSON)')
    parser.add_argument('--implementations', nargs='+',
                      default=['python', 'go', 'typescript'],
                      choices=['python', 'go', 'typescript'],
                      help='Which implementations to run')
    parser.add_argument('--verbose', action='store_true',
                      help='Enable verbose output')
    parser.add_argument('--test-size', type=str, default='xor',
                      choices=['xor', 'small', 'medium', 'large', 'xlarge'],
                      help='Size of the test to run')
    parser.add_argument('--num-threads', type=int, default=4,
                      help='Number of threads for parallel processing')
    
    args = parser.parse_args()
    
    runner = BenchmarkRunner(args)
    runner.run()

if __name__ == "__main__":
    main()
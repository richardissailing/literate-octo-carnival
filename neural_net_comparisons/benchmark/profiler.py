import cProfile
import gc
import pstats
import time
import tracemalloc
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import psutil


@dataclass
class ProfileStats:
    """Container for profiling statistics"""

    execution_time: float
    memory_peak: float
    memory_diff: float
    cpu_percent: float
    function_stats: Dict[str, Any]


class Profiler:
    _active_profilers = set()

    def __init__(self, enabled: bool = True, name: str = "default"):
        self.enabled = enabled
        self.name = name
        self.profiler = None
        self.process = psutil.Process()
        self.start_memory = 0
        self.start_time = 0
        self.stats = None
        self._cleanup_completed = False

    def _cleanup_active_profilers(self):
        """Clean up any active profilers"""
        if not self._cleanup_completed:
            for profiler_id in list(Profiler._active_profilers):
                try:
                    # Disable any active profilers
                    cProfile.Profile().disable()
                except Exception:
                    pass
            Profiler._active_profilers.clear()
            if tracemalloc.is_tracing():
                tracemalloc.stop()
            self._cleanup_completed = True
            gc.collect()

    def __enter__(self):
        if not self.enabled:
            return self

        self._cleanup_active_profilers()

        # Force garbage collection before profiling
        gc.collect()

        # Create a new profiler
        self.profiler = cProfile.Profile()

        try:
            self.profiler.enable()
            Profiler._active_profilers.add(id(self))

            # Start memory tracking
            if not tracemalloc.is_tracing():
                tracemalloc.start()

            # Get initial memory using the correct method name for Python 3.13
            try:
                self.start_memory = getattr(
                    self.process, "get_memory_info"
                )().rss
            except AttributeError:
                self.start_memory = self.process.memory_info().rss

            self.start_time = time.time()

            # Get CPU percentage using the correct method name
            try:
                getattr(self.process, "get_cpu_percent")()
            except AttributeError:
                self.process.cpu_percent()

        except Exception as e:
            self.log(f"Error initializing profiler: {e}")
            self.enabled = False

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.enabled:
            return

        try:
            # Stop CPU profiling
            self.profiler.disable()
            Profiler._active_profilers.discard(id(self))

            # Get final memory measurement
            try:
                current_memory = getattr(self.process, "get_memory_info")().rss
            except AttributeError:
                current_memory = self.process.memory_info().rss

            memory_snapshot = tracemalloc.take_snapshot()

            if len(Profiler._active_profilers) == 0:
                tracemalloc.stop()

            # Calculate stats
            execution_time = time.time() - self.start_time
            memory_diff = current_memory - self.start_memory

            # Get CPU percentage using the correct method
            try:
                cpu_percent = getattr(self.process, "get_cpu_percent")()
            except AttributeError:
                cpu_percent = self.process.cpu_percent()

            # Get function stats
            stats = pstats.Stats(self.profiler)
            stats.sort_stats("cumulative")

            self.stats = ProfileStats(
                execution_time=execution_time,
                memory_peak=memory_snapshot.statistics("filename")[-1].size
                / 1024
                / 1024,  # MB
                memory_diff=memory_diff / 1024 / 1024,  # MB
                cpu_percent=cpu_percent,
                function_stats=dict(stats.stats),
            )

        except Exception as e:
            self.log(f"Error in profiler cleanup: {e}")
            self.stats = None

    def log(self, message: str):
        """Log a message with the profiler name"""
        print(f"[Profiler {self.name}] {message}")

    def save_stats(self, output_dir: Path, name: str):
        """Save profiling statistics to files"""
        if not self.enabled or not self.stats:
            return

        try:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            # Save stats to file
            stats_file = output_dir / f"{name}_profile_stats.prof"
            self.profiler.dump_stats(str(stats_file))

            # Create readable stats
            readable_stats = output_dir / f"{name}_profile_stats.txt"
            with open(readable_stats, "w") as f:
                stats = pstats.Stats(self.profiler, stream=f)
                stats.sort_stats("cumulative")
                stats.print_stats()

                # Add memory and CPU info
                f.write("\n\nMemory and CPU Statistics:\n")
                f.write(
                    f"Execution Time: {self.stats.execution_time:.2f} "
                    "seconds\n"
                )
                f.write(
                    f"Peak Memory Usage: {self.stats.memory_peak:.2f} MB\n"
                )
                f.write(
                    f"Memory Difference: {self.stats.memory_diff:.2f} MB\n"
                )
                f.write(f"CPU Usage: {self.stats.cpu_percent:.1f}%\n")
        except Exception as e:
            self.log(f"Error saving stats: {e}")

    def print_stats(self):
        """Print profiling statistics to console"""
        if not self.enabled or not self.stats:
            self.log("No statistics available")
            return

        try:
            print(f"\nProfiling Results for {self.name}:")
            print("-" * 40)
            print(f"Execution Time: {self.stats.execution_time:.2f} seconds")
            print(f"Peak Memory Usage: {self.stats.memory_peak:.2f} MB")
            print(f"Memory Difference: {self.stats.memory_diff:.2f} MB")
            print(f"CPU Usage: {self.stats.cpu_percent:.1f}%")
            print("\nTop 10 Functions by Cumulative Time:")

            stats = pstats.Stats(self.profiler)
            stats.sort_stats("cumulative")
            stats.print_stats(10)
        except Exception as e:
            self.log(f"Error printing stats: {e}")

import numpy as np
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import List, Callable, Optional, Union

class Matrix:
    def __init__(self, rows: int, cols: int):
        self.rows = rows
        self.cols = cols
        # Use NumPy for efficient storage and operations
        self.data = np.zeros((rows, cols), dtype=np.float32)
    
    @staticmethod
    def from_numpy(data: np.ndarray) -> 'Matrix':
        """Create Matrix from NumPy array"""
        matrix = Matrix(data.shape[0], data.shape[1])
        matrix.data = data.astype(np.float32)
        return matrix
    
    @staticmethod
    def from_list(data: List[List[float]]) -> 'Matrix':
        """Create Matrix from nested list"""
        rows = len(data)
        cols = len(data[0]) if rows > 0 else 0
        matrix = Matrix(rows, cols)
        matrix.data = np.array(data, dtype=np.float32)
        return matrix

    def _multiply_chunk(self, other: 'Matrix', start_row: int, 
                       end_row: int, result: 'Matrix'):
        """Process a chunk of matrix multiplication using NumPy"""
        result.data[start_row:end_row] = np.dot(
            self.data[start_row:end_row], 
            other.data
        )

    def multiply(self, other: 'Matrix', num_threads: int = 4) -> 'Matrix':
        """Matrix multiplication with parallel processing"""
        if self.cols != other.rows:
            raise ValueError("Invalid matrix multiplication dimensions")

        result = Matrix(self.rows, other.cols)
        
        # Use direct NumPy operation for small matrices
        if self.rows * self.cols * other.cols < 10000:
            result.data = np.dot(self.data, other.data)
            return result

        # Parallel processing for larger matrices
        chunk_size = max(1, self.rows // num_threads)
        
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = []
            for start in range(0, self.rows, chunk_size):
                end = min(start + chunk_size, self.rows)
                futures.append(
                    executor.submit(
                        self._multiply_chunk, 
                        other, start, end, result
                    )
                )
            
            # Wait for completion
            for future in futures:
                future.result()

        return result

    def multiply_elementwise(self, other: 'Matrix') -> 'Matrix':
        """Element-wise multiplication (Hadamard product)"""
        if self.rows != other.rows or self.cols != other.cols:
            raise ValueError("Invalid dimensions for elementwise multiplication")
        
        result = Matrix(self.rows, self.cols)
        result.data = self.data * other.data
        return result

    def add(self, other: 'Matrix') -> 'Matrix':
        """Matrix addition"""
        if self.rows != other.rows or self.cols != other.cols:
            raise ValueError("Invalid matrix addition dimensions")
        
        result = Matrix(self.rows, self.cols)
        result.data = self.data + other.data
        return result

    def subtract(self, other: 'Matrix') -> 'Matrix':
        """Matrix subtraction"""
        if self.rows != other.rows or self.cols != other.cols:
            raise ValueError("Invalid matrix subtraction dimensions")
        
        result = Matrix(self.rows, self.cols)
        result.data = self.data - other.data
        return result

    def transpose(self) -> 'Matrix':
        """Matrix transpose"""
        result = Matrix(self.cols, self.rows)
        result.data = self.data.T
        return result

    def map(self, func: Callable[[float], float]) -> 'Matrix':
        """Apply function to each element"""
        result = Matrix(self.rows, self.cols)
        result.data = np.vectorize(func)(self.data)
        return result

    def sum(self, axis: Optional[int] = None) -> Union['Matrix', float]:
        """Sum matrix elements along specified axis"""
        if axis is None:
            return float(np.sum(self.data))
        
        result_data = np.sum(self.data, axis=axis)
        if axis == 0:
            return Matrix.from_numpy(result_data.reshape(1, -1))
        else:
            return Matrix.from_numpy(result_data.reshape(-1, 1))

    def clip(self, min_val: float = -1.0, max_val: float = 1.0) -> 'Matrix':
        """Clip matrix values to given range"""
        result = Matrix(self.rows, self.cols)
        result.data = np.clip(self.data, min_val, max_val)
        return result

    def copy(self) -> 'Matrix':
        """Create a deep copy"""
        result = Matrix(self.rows, self.cols)
        result.data = self.data.copy()
        return result

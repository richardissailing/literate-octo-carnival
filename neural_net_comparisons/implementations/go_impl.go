package main

import (
	"fmt"
	"math"
	"math/rand"
)

type Matrix struct {
	rows, cols int
	data       [][]float64
}

func NewMatrix(rows, cols int) *Matrix {
	m := &Matrix{
		rows: rows,
		cols: cols,
		data: make([][]float64, rows),
	}
	for i := range m.data {
		m.data[i] = make([]float64, cols)
	}
	return m
}

func (m *Matrix) Multiply(other *Matrix) *Matrix {
	if m.cols != other.rows {
		panic("Invalid matrix multiplication dimensions")
	}

	result := NewMatrix(m.rows, other.cols)
	for i := 0; i < m.rows; i++ {
		for j := 0; j < other.cols; j++ {
			sum := 0.0
			for k := 0; k < m.cols; k++ {
				sum += m.data[i][k] * other.data[k][j]
			}
			result.data[i][j] = sum
		}
	}
	return result
}

func (m *Matrix) Add(other *Matrix) *Matrix {
	if m.rows != other.rows || m.cols != other.cols {
		panic("Invalid matrix addition dimensions")
	}

	result := NewMatrix(m.rows, m.cols)
	for i := 0; i < m.rows; i++ {
		for j := 0; j < m.cols; j++ {
			result.data[i][j] = m.data[i][j] + other.data[i][j]
		}
	}
	return result
}

func sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

func sigmoidDerivative(x float64) float64 {
	return x * (1.0 - x)
}

type NeuralNetwork struct {
	layers  []int
	weights []*Matrix
	biases  []*Matrix
}

func NewNeuralNetwork(layers []int) *NeuralNetwork {
	nn := &NeuralNetwork{
		layers:  layers,
		weights: make([]*Matrix, len(layers)-1),
		biases:  make([]*Matrix, len(layers)-1),
	}

	// Initialize weights and biases
	for i := 0; i < len(layers)-1; i++ {
		nn.weights[i] = NewMatrix(layers[i], layers[i+1])
		nn.biases[i] = NewMatrix(1, layers[i+1])

		// He initialization
		scale := math.Sqrt(2.0 / float64(layers[i]))
		for r := 0; r < nn.weights[i].rows; r++ {
			for c := 0; c < nn.weights[i].cols; c++ {
				nn.weights[i].data[r][c] = rand.NormFloat64() * scale
			}
		}
	}

	return nn
}

func (nn *NeuralNetwork) Forward(input *Matrix) []*Matrix {
	activations := make([]*Matrix, len(nn.layers))
	activations[0] = input

	for i := 0; i < len(nn.weights); i++ {
		net := activations[i].Multiply(nn.weights[i])

		// Create expanded bias matrix matching the batch size
		expandedBias := NewMatrix(input.rows, nn.biases[i].cols)
		for r := 0; r < input.rows; r++ {
			for c := 0; c < nn.biases[i].cols; c++ {
				expandedBias.data[r][c] = nn.biases[i].data[0][c]
			}
		}
		net = net.Add(expandedBias)

		activation := NewMatrix(net.rows, net.cols)
		for r := 0; r < net.rows; r++ {
			for c := 0; c < net.cols; c++ {
				activation.data[r][c] = sigmoid(net.data[r][c])
			}
		}
		activations[i+1] = activation
	}

	return activations
}

func calculateAccuracy(predictions *Matrix, targets *Matrix) float64 {
	if predictions.rows != targets.rows || predictions.cols != targets.cols {
		panic("Predictions and targets dimensions don't match")
	}

	correct := 0
	total := predictions.rows

	for i := 0; i < predictions.rows; i++ {
		predicted := 0.0
		if predictions.data[i][0] > 0.5 {
			predicted = 1.0
		}
		if predicted == targets.data[i][0] {
			correct++
		}
	}

	return float64(correct) / float64(total)
}

func (nn *NeuralNetwork) Train(X, y *Matrix, epochs int, learningRate float64) {
	for epoch := 0; epoch < epochs; epoch++ {
		// Forward propagation
		activations := nn.Forward(X)
		output := activations[len(activations)-1]

		// Calculate and print loss every 100 epochs
		if epoch%100 == 0 {
			loss := 0.0
			for i := 0; i < y.rows; i++ {
				diff := output.data[i][0] - y.data[i][0]
				loss += diff * diff
			}
			loss /= float64(y.rows)
			fmt.Printf("Epoch %d, Loss: %.4f\n", epoch, loss)
		}

		// Backward propagation
		m := float64(X.rows)
		delta := NewMatrix(output.rows, output.cols)
		for i := 0; i < output.rows; i++ {
			for j := 0; j < output.cols; j++ {
				delta.data[i][j] = output.data[i][j] - y.data[i][j]
			}
		}

		for i := len(nn.weights) - 1; i >= 0; i-- {
			activation := activations[i]

			// Calculate gradients
			gradW := NewMatrix(nn.weights[i].rows, nn.weights[i].cols)
			gradB := NewMatrix(1, nn.biases[i].cols)

			for r := 0; r < gradW.rows; r++ {
				for c := 0; c < gradW.cols; c++ {
					sum := 0.0
					for k := 0; k < delta.rows; k++ {
						sum += activation.data[k][r] * delta.data[k][c]
					}
					gradW.data[r][c] = sum / m
				}
			}

			for c := 0; c < gradB.cols; c++ {
				sum := 0.0
				for r := 0; r < delta.rows; r++ {
					sum += delta.data[r][c]
				}
				gradB.data[0][c] = sum / m
			}

			// Update weights and biases
			for r := 0; r < nn.weights[i].rows; r++ {
				for c := 0; c < nn.weights[i].cols; c++ {
					nn.weights[i].data[r][c] -= learningRate * gradW.data[r][c]
				}
			}

			for c := 0; c < nn.biases[i].cols; c++ {
				nn.biases[i].data[0][c] -= learningRate * gradB.data[0][c]
			}

			// Calculate delta for next layer if not input layer
			if i > 0 {
				newDelta := NewMatrix(delta.rows, activation.cols)
				for r := 0; r < delta.rows; r++ {
					for c := 0; c < activation.cols; c++ {
						sum := 0.0
						for k := 0; k < delta.cols; k++ {
							sum += delta.data[r][k] * nn.weights[i].data[c][k]
						}
						newDelta.data[r][c] = sum * sigmoidDerivative(activation.data[r][c])
					}
				}
				delta = newDelta
			}
		}
	}

	// Calculate and print final accuracy
	final_predictions := nn.Forward(X)[len(nn.layers)-1]
	accuracy := calculateAccuracy(final_predictions, y)
	fmt.Printf("Final Accuracy: %.4f\n", accuracy)
}

func main() {
	// Set random seed for reproducibility
	rand.Seed(42)

	// XOR problem
	X := NewMatrix(4, 2)
	X.data = [][]float64{
		{0, 0},
		{0, 1},
		{1, 0},
		{1, 1},
	}

	y := NewMatrix(4, 1)
	y.data = [][]float64{
		{0},
		{1},
		{1},
		{0},
	}

	// Create neural network with architecture: 2-4-1
	nn := NewNeuralNetwork([]int{2, 4, 1})

	// Train the network
	nn.Train(X, y, 1000, 0.1)

	// Print predictions
	predictions := nn.Forward(X)[len(nn.layers)-1]
	fmt.Println("\nPredictions:")
	for i := 0; i < X.rows; i++ {
		fmt.Printf("Input: [%.0f %.0f], Target: %.0f, Predicted: %.4f\n",
			X.data[i][0], X.data[i][1], y.data[i][0], predictions.data[i][0])
	}
}

class Matrix {
    rows: number;
    cols: number;
    data: number[][];

    constructor(rows: number, cols: number) {
        this.rows = rows;
        this.cols = cols;
        this.data = Array(rows).fill(0).map(() => Array(cols).fill(0));
    }

    multiply(other: Matrix): Matrix {
        if (this.cols !== other.rows) {
            throw new Error("Invalid matrix multiplication dimensions");
        }

        const result = new Matrix(this.rows, other.cols);
        for (let i = 0; i < this.rows; i++) {
            for (let j = 0; j < other.cols; j++) {
                let sum = 0;
                for (let k = 0; k < this.cols; k++) {
                    sum += this.data[i][k] * other.data[k][j];
                }
                result.data[i][j] = sum;
            }
        }
        return result;
    }

    add(other: Matrix): Matrix {
        if (this.rows !== other.rows || this.cols !== other.cols) {
            throw new Error("Invalid matrix addition dimensions");
        }

        const result = new Matrix(this.rows, this.cols);
        for (let i = 0; i < this.rows; i++) {
            for (let j = 0; j < this.cols; j++) {
                result.data[i][j] = this.data[i][j] + other.data[i][j];
            }
        }
        return result;
    }

    static subtract(a: Matrix, b: Matrix): Matrix {
        if (a.rows !== b.rows || a.cols !== b.cols) {
            throw new Error("Invalid matrix subtraction dimensions");
        }

        const result = new Matrix(a.rows, a.cols);
        for (let i = 0; i < a.rows; i++) {
            for (let j = 0; j < a.cols; j++) {
                result.data[i][j] = a.data[i][j] - b.data[i][j];
            }
        }
        return result;
    }

    transpose(): Matrix {
        const result = new Matrix(this.cols, this.rows);
        for (let i = 0; i < this.rows; i++) {
            for (let j = 0; j < this.cols; j++) {
                result.data[j][i] = this.data[i][j];
            }
        }
        return result;
    }

    map(func: (x: number) => number): Matrix {
        const result = new Matrix(this.rows, this.cols);
        for (let i = 0; i < this.rows; i++) {
            for (let j = 0; j < this.cols; j++) {
                result.data[i][j] = func(this.data[i][j]);
            }
        }
        return result;
    }
}

class NeuralNetwork {
    layers: number[];
    weights: Matrix[];
    biases: Matrix[];
    activations: Matrix[];

    constructor(layers: number[]) {
        this.layers = layers;
        this.weights = [];
        this.biases = [];
        this.activations = [];

        // Initialize weights and biases
        for (let i = 0; i < layers.length - 1; i++) {
            const w = new Matrix(layers[i], layers[i + 1]);
            const b = new Matrix(1, layers[i + 1]);

            // He initialization
            const scale = Math.sqrt(2.0 / layers[i]);
            for (let r = 0; r < w.rows; r++) {
                for (let c = 0; c < w.cols; c++) {
                    w.data[r][c] = (Math.random() * 2 - 1) * scale;
                }
            }

            this.weights.push(w);
            this.biases.push(b);
        }
    }

    sigmoid(x: number): number {
        return 1 / (1 + Math.exp(-x));
    }

    sigmoidDerivative(x: number): number {
        return x * (1 - x);
    }

    forward(input: Matrix): Matrix[] {
        this.activations = [input];

        for (let i = 0; i < this.weights.length; i++) {
            const net = this.activations[i].multiply(this.weights[i]);
            
            // Create expanded bias matrix matching the batch size
            const expandedBias = new Matrix(input.rows, this.biases[i].cols);
            for (let r = 0; r < input.rows; r++) {
                for (let c = 0; c < this.biases[i].cols; c++) {
                    expandedBias.data[r][c] = this.biases[i].data[0][c];
                }
            }
            const netWithBias = net.add(expandedBias);
            this.activations.push(netWithBias.map(x => this.sigmoid(x)));
        }

        return this.activations;
    }

    train(X: Matrix, y: Matrix, epochs: number = 1000, learningRate: number = 0.1): void {
        const m = X.rows;

        for (let epoch = 0; epoch < epochs; epoch++) {
            // Forward propagation
            const activations = this.forward(X);
            const output = activations[activations.length - 1];

            // Calculate and log loss every 100 epochs
            if (epoch % 100 === 0) {
                let loss = 0;
                for (let i = 0; i < y.rows; i++) {
                    for (let j = 0; j < y.cols; j++) {
                        const diff = output.data[i][j] - y.data[i][j];
                        loss += diff * diff;
                    }
                }
                loss /= m;
                console.log(`Epoch ${epoch}, Loss: ${loss.toFixed(4)}`);
            }

            // Backward propagation
            let delta = Matrix.subtract(output, y);

            for (let i = this.weights.length - 1; i >= 0; i--) {
                const activation = this.activations[i];

                // Calculate gradients
                const weightGrad = activation.transpose().multiply(delta);
                const biasGrad = delta;

                // Update weights and biases
                for (let r = 0; r < this.weights[i].rows; r++) {
                    for (let c = 0; c < this.weights[i].cols; c++) {
                        this.weights[i].data[r][c] -= learningRate * weightGrad.data[r][c] / m;
                    }
                }

                for (let c = 0; c < this.biases[i].cols; c++) {
                    let sum = 0;
                    for (let r = 0; r < delta.rows; r++) {
                        sum += delta.data[r][c];
                    }
                    this.biases[i].data[0][c] -= learningRate * sum / m;
                }

                if (i > 0) {
                    delta = delta.multiply(this.weights[i].transpose()).map(x => this.sigmoidDerivative(x));
                }
            }
        }

        // Calculate final accuracy
        const predictions = this.forward(X)[this.layers.length - 1];
        let correct = 0;
        for (let i = 0; i < X.rows; i++) {
            const predicted = predictions.data[i][0] > 0.5 ? 1 : 0;
            if (predicted === y.data[i][0]) {
                correct++;
            }
        }
        const accuracy = correct / X.rows;
        console.log(`Final Accuracy: ${accuracy.toFixed(4)}`);
    }
}

// Example usage
const X = new Matrix(4, 2);
X.data = [[0, 0], [0, 1], [1, 0], [1, 1]];

const y = new Matrix(4, 1);
y.data = [[0], [1], [1], [0]];

// Create and train network
const nn = new NeuralNetwork([2, 4, 1]);
nn.train(X, y, 1000, 0.1);

// Print predictions
const predictions = nn.forward(X)[nn.layers.length - 1];
console.log("\nPredictions:");
for (let i = 0; i < X.rows; i++) {
    console.log(`Input: [${X.data[i][0]} ${X.data[i][1]}], Target: ${y.data[i][0]}, Predicted: ${predictions.data[i][0].toFixed(4)}`);
}
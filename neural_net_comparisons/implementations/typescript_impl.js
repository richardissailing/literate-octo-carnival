var Matrix = /** @class */ (function () {
    function Matrix(rows, cols) {
        this.rows = rows;
        this.cols = cols;
        this.data = Array(rows).fill(0).map(function () { return Array(cols).fill(0); });
    }
    Matrix.prototype.multiply = function (other) {
        if (this.cols !== other.rows) {
            throw new Error("Invalid matrix multiplication dimensions");
        }
        var result = new Matrix(this.rows, other.cols);
        for (var i = 0; i < this.rows; i++) {
            for (var j = 0; j < other.cols; j++) {
                var sum = 0;
                for (var k = 0; k < this.cols; k++) {
                    sum += this.data[i][k] * other.data[k][j];
                }
                result.data[i][j] = sum;
            }
        }
        return result;
    };
    Matrix.prototype.add = function (other) {
        if (this.rows !== other.rows || this.cols !== other.cols) {
            throw new Error("Invalid matrix addition dimensions");
        }
        var result = new Matrix(this.rows, this.cols);
        for (var i = 0; i < this.rows; i++) {
            for (var j = 0; j < this.cols; j++) {
                result.data[i][j] = this.data[i][j] + other.data[i][j];
            }
        }
        return result;
    };
    Matrix.subtract = function (a, b) {
        if (a.rows !== b.rows || a.cols !== b.cols) {
            throw new Error("Invalid matrix subtraction dimensions");
        }
        var result = new Matrix(a.rows, a.cols);
        for (var i = 0; i < a.rows; i++) {
            for (var j = 0; j < a.cols; j++) {
                result.data[i][j] = a.data[i][j] - b.data[i][j];
            }
        }
        return result;
    };
    Matrix.prototype.transpose = function () {
        var result = new Matrix(this.cols, this.rows);
        for (var i = 0; i < this.rows; i++) {
            for (var j = 0; j < this.cols; j++) {
                result.data[j][i] = this.data[i][j];
            }
        }
        return result;
    };
    Matrix.prototype.map = function (func) {
        var result = new Matrix(this.rows, this.cols);
        for (var i = 0; i < this.rows; i++) {
            for (var j = 0; j < this.cols; j++) {
                result.data[i][j] = func(this.data[i][j]);
            }
        }
        return result;
    };
    return Matrix;
}());
var NeuralNetwork = /** @class */ (function () {
    function NeuralNetwork(layers) {
        this.layers = layers;
        this.weights = [];
        this.biases = [];
        this.activations = [];
        // Initialize weights and biases
        for (var i = 0; i < layers.length - 1; i++) {
            var w = new Matrix(layers[i], layers[i + 1]);
            var b = new Matrix(1, layers[i + 1]);
            // He initialization
            var scale = Math.sqrt(2.0 / layers[i]);
            for (var r = 0; r < w.rows; r++) {
                for (var c = 0; c < w.cols; c++) {
                    w.data[r][c] = (Math.random() * 2 - 1) * scale;
                }
            }
            this.weights.push(w);
            this.biases.push(b);
        }
    }
    NeuralNetwork.prototype.sigmoid = function (x) {
        return 1 / (1 + Math.exp(-x));
    };
    NeuralNetwork.prototype.sigmoidDerivative = function (x) {
        return x * (1 - x);
    };
    NeuralNetwork.prototype.forward = function (input) {
        var _this = this;
        this.activations = [input];
        for (var i = 0; i < this.weights.length; i++) {
            var net = this.activations[i].multiply(this.weights[i]);
            // Create expanded bias matrix matching the batch size
            var expandedBias = new Matrix(input.rows, this.biases[i].cols);
            for (var r = 0; r < input.rows; r++) {
                for (var c = 0; c < this.biases[i].cols; c++) {
                    expandedBias.data[r][c] = this.biases[i].data[0][c];
                }
            }
            var netWithBias = net.add(expandedBias);
            this.activations.push(netWithBias.map(function (x) { return _this.sigmoid(x); }));
        }
        return this.activations;
    };
    NeuralNetwork.prototype.train = function (X, y, epochs, learningRate) {
        var _this = this;
        if (epochs === void 0) { epochs = 1000; }
        if (learningRate === void 0) { learningRate = 0.1; }
        var m = X.rows;
        for (var epoch = 0; epoch < epochs; epoch++) {
            // Forward propagation
            var activations = this.forward(X);
            var output = activations[activations.length - 1];
            // Calculate and log loss every 100 epochs
            if (epoch % 100 === 0) {
                var loss = 0;
                for (var i = 0; i < y.rows; i++) {
                    for (var j = 0; j < y.cols; j++) {
                        var diff = output.data[i][j] - y.data[i][j];
                        loss += diff * diff;
                    }
                }
                loss /= m;
                console.log("Epoch ".concat(epoch, ", Loss: ").concat(loss.toFixed(4)));
            }
            // Backward propagation
            var delta = Matrix.subtract(output, y);
            for (var i = this.weights.length - 1; i >= 0; i--) {
                var activation = this.activations[i];
                // Calculate gradients
                var weightGrad = activation.transpose().multiply(delta);
                var biasGrad = delta;
                // Update weights and biases
                for (var r = 0; r < this.weights[i].rows; r++) {
                    for (var c = 0; c < this.weights[i].cols; c++) {
                        this.weights[i].data[r][c] -= learningRate * weightGrad.data[r][c] / m;
                    }
                }
                for (var c = 0; c < this.biases[i].cols; c++) {
                    var sum = 0;
                    for (var r = 0; r < delta.rows; r++) {
                        sum += delta.data[r][c];
                    }
                    this.biases[i].data[0][c] -= learningRate * sum / m;
                }
                if (i > 0) {
                    delta = delta.multiply(this.weights[i].transpose()).map(function (x) { return _this.sigmoidDerivative(x); });
                }
            }
        }
        // Calculate final accuracy
        var predictions = this.forward(X)[this.layers.length - 1];
        var correct = 0;
        for (var i = 0; i < X.rows; i++) {
            var predicted = predictions.data[i][0] > 0.5 ? 1 : 0;
            if (predicted === y.data[i][0]) {
                correct++;
            }
        }
        var accuracy = correct / X.rows;
        console.log("Final Accuracy: ".concat(accuracy.toFixed(4)));
    };
    return NeuralNetwork;
}());
// Example usage
var X = new Matrix(4, 2);
X.data = [[0, 0], [0, 1], [1, 0], [1, 1]];
var y = new Matrix(4, 1);
y.data = [[0], [1], [1], [0]];
// Create and train network
var nn = new NeuralNetwork([2, 4, 1]);
nn.train(X, y, 1000, 0.1);
// Print predictions
var predictions = nn.forward(X)[nn.layers.length - 1];
console.log("\nPredictions:");
for (var i = 0; i < X.rows; i++) {
    console.log("Input: [".concat(X.data[i][0], " ").concat(X.data[i][1], "], Target: ").concat(y.data[i][0], ", Predicted: ").concat(predictions.data[i][0].toFixed(4)));
}

import numpy as np
from timeit import default_timer as timer
from tqdm.auto import tqdm


class NeuralNetwork:
    def __init__(self, layers):
        '''
        Initializes the neural network with specified layer sizes
        '''
        self.layers = layers
        self.num_layers = len(layers)
        self.weights = []
        self.biases = []

        for i in range(self.num_layers - 1):
            # He initialization (W ~ N(0, sqrt(2/n)))
            weight = np.random.randn(layers[i], layers[i + 1]) * np.sqrt(2 / layers[i])
            # Biases start at zero
            bias = np.zeros((1, layers[i+1]))
            self.weights.append(weight)
            self.biases.append(bias)

    def relu(self, x):
        '''
        The ReLU activation function. Intended to help for non-linear modeling
        '''
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        '''
        Take derivative of the ReLU. Required for the backpropagation
        '''
        return (x > 0).astype(float)

    def softmax(self, z):
        '''
        Softmax activation
        '''
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    def forward(self, X):
        '''
        Performs the forward propagation
        '''
        self.activation = [X]
        self.z_vals = []

        for i in range(self.num_layers - 1):
            z = np.dot(self.activation[-1], self.weights[i]) + self.biases[i]
            self.z_vals.append(z)

            # Hidden layers
            if i < self.num_layers - 2:
                a = self.relu(z)
            else:
                a = self.softmax(z)
            
            self.activation.append(a)
        
        return self.activation[-1]

    def backward(self, X, y, lr):
        '''
        Performs the backpropagation
        '''
        m = X.shape[0]
        error = self.activation[-1] - y

        for i in range(self.num_layers - 2, -1, -1):
            change_weight = np.dot(self.activation[i].T, error) / m
            change_bias = np.sum(error, axis=0, keepdims=True) / m

            self.weights[i] -= change_weight * lr
            self.biases[i] -= change_bias * lr

            if i > 0:
                error = np.dot(error, self.weights[i].T) * self.relu_derivative(self.z_vals[i-1])

    def compute_cross_entropy_loss(self, y_pred, y_true):
        '''
        Calculates the cross entropy loss
        '''
        n = y_true.shape[0]
        log_y_true = np.log(y_pred[range(n), y_true.argmax(axis=1)])
        sum = np.sum(-1 * log_y_true) / n
        return sum

    def predict(self, X):
        '''
        Uses the argmax function to make predicitions that'll be used in the evaluation accuracy of our model
        '''
        return np.argmax(self.forward(X), axis=1)
    
    def evaluate_accuracy(self, X, y):
        '''
        Gives the accuracy of our predicted values to true values. Considered the testing step of our machine learning model
        '''
        predictions = self.predict(X)
        labels = np.argmax(y, axis=1)
        accuracy = np.mean(predictions == labels)
        return accuracy

    def print_train_time(self,
                         start: float,
                         end: float):
        '''
        Calculates the train time when given start and end times.
        '''
        total_time = end - start
        print(f"Train time: {total_time:.3f} seconds")
        return total_time

    def train_model(self, X_train, y_train, X_val, y_val, epochs = 15, batch_size=32, lr=0.01):
        '''
        Runs the training loop of our model.
        '''
        train_time_start = timer()

        training_loss_array = []
        accuracy_array = []

        for epoch in tqdm(range(epochs)):
            indices = np.random.permutation(X_train.shape[0])

            randomized_X = X_train[indices]
            randomized_y = y_train[indices]

            for i in range(0, X_train.shape[0], batch_size):
                X_batch = randomized_X[i:batch_size+i]
                y_batch = randomized_y[i:batch_size+i]

                self.forward(X_batch)
                self.backward(X_batch, y_batch, lr)

            train_pred = self.forward(X_train)
            
            train_loss = self.compute_cross_entropy_loss(train_pred, y_train)
            training_loss_array.append(train_loss)

            accuracy = self.evaluate_accuracy(X_val, y_val)
            accuracy_array.append(accuracy)

            print(f"Epoch: {epoch+1}/{epochs} | Loss: {train_loss:.4f} | Accuracy: {accuracy:.4f}")
        
        train_time_end = timer()
        self.print_train_time(train_time_start, train_time_end)

        return training_loss_array, accuracy_array
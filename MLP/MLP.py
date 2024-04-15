import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_der(x):
    return x * (1 - x)

class MultilayerPerceptron:
    def __init__(self, train_data, target, n_iter=100, num_input=2, num_hidden=2, num_output=1, lr=0.1):
        self.training_data = train_data
        self.target_data = target
        self.n_iter = n_iter
        self.lr = lr
        
        # Initialize weights randomly
        self.input_hidden_weights = np.random.uniform(size=(num_input, num_hidden))
        self.hidden_output_weights = np.random.uniform(size=(num_hidden, num_output))
        self.hidden_bias = np.random.uniform(size=(1,num_hidden))
        self.output_bias = np.random.uniform(size=(1,num_output))

        # Initialize plot lists
        self.mse = []
        self.hidden_mse = []
        self.classification_errors = []
        self.input_hidden_weights_history = []
        self.hidden_output_weights_history = []

    def update_weights(self):
        # Calculate predition error
        prediction_error = self.target_data - self.output_final
        # Calculate MSE
        self.mse.append(np.mean(np.square(prediction_error)))
        # Calculate MSE on the hidden layer
        hidden_prediction_error = np.dot(prediction_error, self.hidden_output_weights.T)
        hidden_mse = np.mean(np.square(hidden_prediction_error))
        self.hidden_mse.append(hidden_mse)
        # Calculate the gradients of the weights connecting the input layer to the hidden layer
        input_hidden_weight_gradients = np.dot(self.training_data.T, (((prediction_error * sigmoid_der(self.output_final)) * self.hidden_output_weights.T) * sigmoid_der(self.hidden_output)))
        # Calculate the gradients of the weights connecting the hidden layer to the output layer
        hidden_output_weight_gradients = np.dot(self.hidden_output.T, (prediction_error * sigmoid_der(self.output_final)))
        # Update the weights connecting the input layer to the hidden layer
        self.input_hidden_weights += self.lr * input_hidden_weight_gradients
        # Update the weights connecting the hidden layer to the output layer
        self.hidden_output_weights += self.lr * hidden_output_weight_gradients
        # Update the biases of the neurons in the hidden layer
        self.hidden_bias += np.sum(self.lr * ((prediction_error * sigmoid_der(self.output_final)) * self.hidden_output_weights.T) * sigmoid_der(self.hidden_output), axis=0)
        # Update the biases of the neurons in the output layer
        self.output_bias += np.sum(self.lr * prediction_error * sigmoid_der(self.output_final), axis=0)
        # Append current weights to history
        self.input_hidden_weights_history.append(self.input_hidden_weights.copy())
        self.hidden_output_weights_history.append(self.hidden_output_weights.copy())

    def forward(self, input_data):
        # Calculate the input to the hidden layer
        self.hidden_input = np.dot(input_data, self.input_hidden_weights) + self.hidden_bias
        # Apply the sigmoid activation function to the hidden layer input
        self.hidden_output = sigmoid(self.hidden_input)
        # Calculate the input to the output layer
        self.output_input = np.dot(self.hidden_output, self.hidden_output_weights) + self.output_bias
        # Apply the sigmoid activation function to the output layer input
        self.output_final = sigmoid(self.output_input)
        return self.output_final

    def classify(self, point):
        point = np.transpose(point)
        output = self.forward(point)
        return 1 if output >= 0.5 else 0
    
    def train(self):
        for _ in range(self.n_iter):
            self.forward(self.training_data)
            self.update_weights()
            errors = 0
            for i in range(len(self.training_data)):
                if self.classify(self.training_data[i]) != self.target_data[i]:
                    errors += 1
            self.classification_errors.append(errors)

    def test(self, X_test_data):
        y_prediction = self.forward(X_test_data)
        print("Input|Output")
        for i in range(len(X_test)):
            print(f"{X_test_data[i]}|{y_prediction[i]}")

    def plot_all_mse(self):
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 2, 1)
        plt.plot(range(len(self.mse)), self.mse)
        plt.title('Mean Squared Error (MSE) Over Iters - Output')
        plt.xlabel('Iters')
        plt.ylabel('MSE')
        plt.subplot(1, 2, 2)
        plt.plot(range(len(self.hidden_mse)), self.hidden_mse)
        plt.title('Mean Squared Error (MSE) Over Iters - Hidden Layer')
        plt.xlabel('Iters')
        plt.ylabel('MSE')
        plt.grid()
        plt.show()
        
    def plot_classification_errors(self):
        plt.plot(range(len(self.classification_errors)), self.classification_errors)
        plt.title('Classification Errors Over Iters')
        plt.xlabel('Iters')
        plt.ylabel('Misclassified Points')
        plt.grid()
        plt.show()

    def plot_weights(self):
        plt.subplot(1, 2, 1)
        for i in range(len(self.input_hidden_weights_history[0])):
            for j in range(len(self.input_hidden_weights_history[0][0])):
                weights = [iter[i][j] for iter in self.input_hidden_weights_history]
                plt.plot(range(len(weights)), weights, label=f'Input {i+1} to Hidden {j+1}')
        plt.title('Input-Hidden Weights History')
        plt.xlabel('Iters')
        plt.ylabel('Weights')
        plt.legend()
        plt.subplot(1, 2, 2)
        for i in range(len(self.hidden_output_weights_history[0])):
            for j in range(len(self.hidden_output_weights_history[0][0])):
                weights = [iter[i][j] for iter in self.hidden_output_weights_history]
                plt.plot(range(len(weights)), weights, label=f'Hidden {i+1} to Output {j+1}')
        plt.title('Hidden-Output Weights History')
        plt.xlabel('Iters')
        plt.ylabel('Weights')
        plt.grid()
        plt.legend()
        plt.show()

if __name__ == "__main__":
    
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])

    mlp = MultilayerPerceptron(train_data=X, target=y, n_iter=10000)
    mlp.train()
    
    X_test = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    mlp.test(X_test)

    mlp.plot_classification_errors()
    mlp.plot_weights()
    mlp.plot_all_mse()

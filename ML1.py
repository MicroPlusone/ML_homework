import numpy as np

class Homework1_zjy:
    def __init__(self, input_size, learning_rate=0.1, max_epochs=100, random_seed=None):
        self.input_size = input_size
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.random_seed = random_seed
        self.weights = None
        self.bias = None

    def initialize_weights(self):
        if self.random_seed:
            np.random.seed(self.random_seed)
        #self.random_seed=1000
        # weight is random, very good!
        self.weights = np.random.randn(self.input_size)
        self.bias = np.random.randn()

    def activation_function(self, x):
        # active is good
        return 1 if x >= 0 else 0

    def predict(self, x):
        # calculate output(data)
        weighted_sum = np.dot(x, self.weights) + self.bias
        return self.activation_function(weighted_sum)

    def train(self, X, y):
        # setup initial
        self.initialize_weights()
        # censor
        for epoch in range(self.max_epochs):
            total_error = 0
            for i in range(X.shape[0]):
                # predict
                prediction = self.predict(X[i])
                # error
                error = y[i] - prediction
                # update
                self.weights += self.learning_rate * error * X[i]
                self.bias += self.learning_rate * error
                # 累计误差
                total_error += error ** 2
            # convergence judgement
            if total_error == 0:
                print(f"Training converged epoch :{epoch+1}")
                break

def main():
    # setup, oringin, initial data
    X = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]])
    y = np.array([0, 1, 1, 0])

    # train
    perceptron = Homework1_zjy(input_size=3, learning_rate=0.1, max_epochs=100, random_seed=42)
    perceptron.train(X, y)

    # output
    print("\nZero-error combinations:")
    zero_error_combinations = []
    for a in range(2):
        for b in range(2):
            for c in range(2):
                for d in range(2):
                    input_data = np.array([a, b, c])
                    expected_output = d
                    if perceptron.predict(input_data) == expected_output:
                        zero_error_combinations.append((a, b, c, d))

    for combination in zero_error_combinations:
        print(combination)

    print("\nTotal :", len(zero_error_combinations))

if __name__ == "__main__":
    main()

#This homework is oringined by MMinuzero by himself. I swear!

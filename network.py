import numpy as np

X = [[1, 2, 3, 2.5],
     [2.0, 5.0, -1.0, 2.0],
     [-1.5, 2.7, 3.3, -0.8]]



class Layer_Dense:

    def __init__(self, n_inputs, n_nuerons):

        self.weights = np.random.randn(n_inputs, n_nuerons)

        self.biases = np.zeros((1, n_nuerons))

    def forward(self, inputs):

        self.output = np.dot(inputs, self.weights) + self.biases
        self.inputs = inputs

    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dinputs = np.dot(dvalues, self.weights.T)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)



class Activation_reLU:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)


    def backward(self, dvalues):
        self.dinputs = dvalues.copy()

        self.dinputs[dvalues <= 0] = 0


class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        norm_values = exp_values / np.sum(exp_values, axis=1, keepdims=True)

        self.output = norm_values


class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)

        return data_loss

class Loss_CatergoricalLossEntropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]

        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped*y_true, axis=1)

        negative_log_likelihoods = -np.log(correct_confidences)

        return negative_log_likelihoods


layer1 = Layer_Dense(2,3)
activation1 = Activation_reLU()

layer2 = Layer_Dense(3,3)
activation2 = Activation_Softmax()

layer1.forward(X)
activation1.forward(layer1.output)

layer2.forward(activation1.output)
activation2.forward(layer2.output)

print(activation2.output[:5])

loss_function = Loss_CatergoricalLossEntropy()
loss = loss_function.calculate(activation2.output, y)
print(f"Loss: {loss}")


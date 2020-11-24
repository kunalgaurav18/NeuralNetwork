import numpy as np
from NeuralNetwork.neuron import Neuron



class NeuralNetwork:
    def __init__(self, n_input, n_hidden, n_output):
        self.input_layer = []
        self.output_layer = []
        self.hidden_layer = []
        self.learning_rate = 0.8
        self.alpha = 0.1
        self.lmbda = 0.8
        self.init_network(n_input, n_hidden, n_output)

    def init_network(self, n_input, n_hidden, n_output):
        for i in range(n_input):
            self.input_layer.append(Neuron(0))
        self.input_layer.append(Neuron(0, True))
        for i in range(n_hidden):
            self.hidden_layer.append(Neuron(n_input + 1))
        self.hidden_layer.append(Neuron(0, True))
        for i in range(n_output):
            self.output_layer.append(Neuron(n_hidden + 1))

    def feed_forward(self, x, y):
        if x.shape[1] != len(self.input_layer) - 1:
            raise Exception('Check input shape')
        if y.shape[1] != len(self.output_layer):
            raise Exception('Check output shape')
        # calculation of h
        for inputs, outputs in zip(x,y):
            h = np.ones([len(self.hidden_layer)])
            inputs = np.append(inputs, 1)
            for i, hn in enumerate(self.hidden_layer[:-1]):
                hn.input_value = np.dot(hn.incoming_weights,inputs.T)
                hn.activation_value = self.sigmoid(hn.input_value)
                h[i] = hn.activation_value
            for k, on in enumerate(self.output_layer):
                on.input_value = np.dot(on.incoming_weights, h.T)
                on.activation_value = on.input_value
                on.actual_value = outputs[k]
                on.errors = np.hstack((on.errors if hasattr(on, 'errors') else np.array([]),
                                       np.array([on.actual_value - on.activation_value])))

    def back_propagation(self):
        pass

    def sigmoid(self, v, derivative=False):
        if (derivative == True):
            return self.sigmoid(v) * (1 - self.sigmoid(v))
        return 1/1+np.exp(-self.lmbda * v)

    def __str__(self):
        print('-------Input Layer------------')
        for i in self.input_layer:
            print(i)
        print('===============================')
        print('-------Hidden Layer------------')
        for i in self.hidden_layer:
            print(i)
        print('===============================')
        print('-------Output Layer------------')
        for i in self.output_layer:
            print(i)
        print('===============================')
        return ''

    def summary(self):
        print('=======Model Summary==========')
        print('Layer name \t Neurons with bias')
        print(f'Input Layer \t {len(self.input_layer)}')
        print(f'Hidden Layer \t {len(self.hidden_layer)}')
        print(f'Output Layer \t {len(self.output_layer)}')
        print('===============================')



if __name__ == '__main__':
    a = NeuralNetwork(2,10,2)
    a.summary()
    print(a)
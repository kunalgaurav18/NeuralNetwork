import numpy as np

class Neuron:
    def __init__(self, n_connections, isbias = False):
        self.input_value = 0.0
        self.activation_value = 0.0
        self.n_connections = n_connections
        self.incoming_weights = np.random.uniform(low=-1, high=1, size=n_connections)
        if isbias:
            self.input_value = 1.0
            self.activation_value = 1.0



    def __str__(self):
        string = self.__dict__
        return str(string)


if __name__ == '__main__':
    a = Neuron(10)
    print(a)
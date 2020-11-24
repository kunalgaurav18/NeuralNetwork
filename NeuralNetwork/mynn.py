import numpy as np
import os
np.set_printoptions(suppress=True)
np.random.seed(1)

class MyNN:
    def __init__(self):
        self.i2h_weights = None
        self.h2o_weights = None
        self.lmbda = 0.6
        self.alpha = 0.2
        self.eta = 0.4
        self.error = []
        self.output_delta_weight = 0.0
        self.hidden_delta_weight = 0.0
        self.max = None
        self.min = None

    def init_network(self, *args):
        if len(args) != 3:
            raise ValueError('Exactly 3 arguments required: input_nodes, hidden_nodes, output_nodes')
        input_nodes = args[0]
        hidden_nodes = args[1]
        output_nodes = args[2]
        self.i2h_weights = np.random.uniform(low=-1, high=1, size=(hidden_nodes, input_nodes+1))
        self.h2o_weights = np.random.uniform(low=-1, high=1, size=(output_nodes, hidden_nodes+1))


    def feed_forward(self, x, y, predict=False):
        x = np.append(x, 1)
        hidden_iv = self.matrix_multiplication(self.i2h_weights, x.reshape(-1, 1))
        # print('i2h weights: ', self.i2h_weights)
        # print('x: ',x.reshape(-1,1))
        # print('hidden iv: ', hidden_iv)
        hidden_av = self.sigmoid(hidden_iv)
        # print('hidden av: ', hidden_av)
        hidden_av = np.append(hidden_av, 1)
        output_iv = self.matrix_multiplication(self.h2o_weights, hidden_av.reshape(-1, 1))
        # print('output iv: ', output_iv)
        output_av = self.sigmoid(output_iv)
        # print('output av: ', output_av)
        # print('actual op: ', y.reshape(-1, 1))
        # print('error e: ', e)
        e = y.reshape(-1, 1) - output_av
        self.error.append(np.square(e).mean())
        if predict:
            return output_av, e
        self.back_propagation(hidden_av, output_av, e, x)


    def output_gradient(self, v, e):
        return self.sigmoid(v, derivative=True) * e

    def hidden_gradient(self, h, grad_output):
        return self.sigmoid(h.reshape(-1, 1), derivative=True) * self.matrix_multiplication(self.h2o_weights.T, grad_output)

    def delta_weight_output(self, grad_output, hidden_av, prev_delta_weight):
        return self.eta * self.matrix_multiplication(grad_output, hidden_av.reshape(-1, 1).T) + self.alpha * prev_delta_weight

    def delta_weight_hidden(self, grad_hidden, x_arr, prev_delta_weight):
        # if not self.hidden_delta_weight:
        #     prev_delta = 0.0
        # else:
        #     prev_delta = self.hidden_delta_weight[-1]
        return self.eta * self.matrix_multiplication(grad_hidden, x_arr.reshape(-1, 1).T) + self.alpha * prev_delta_weight

    def back_propagation(self, hidden_av, output_av, e, x):
        grad_output = self.output_gradient(output_av, e)
        grad_hidden = self.hidden_gradient(hidden_av, grad_output)
        self.output_delta_weight = self.delta_weight_output(grad_output, hidden_av, self.output_delta_weight)
        self.hidden_delta_weight = self.delta_weight_hidden(grad_hidden, x, self.hidden_delta_weight)
        # Updating weight
        # print('old output wt: ', self.h2o_weights)
        # print('old hidden wt: ', self.i2h_weights)
        self.h2o_weights = self.h2o_weights + self.output_delta_weight
        self.i2h_weights = self.i2h_weights + self.hidden_delta_weight[:-1,:]
        # print('new output wt: ', self.h2o_weights)
        # print('new hidden wt: ', self.i2h_weights)
        # print('error  :', self.error)
        # print('grad output: ', grad_output)
        # print('grad hidden: ', grad_hidden)
        # print('odw: ', self.output_delta_weight)
        # print('hdw: ', self.hidden_delta_weight)

    def fit(self, inputs, outputs, epoch):
        for t in range(epoch):
            for i, o in zip(inputs, outputs):
                self.feed_forward(i, o)
            print('. ', t)
        print('Printing final error-------------------------------')
        print('length of error: ', len(self.error))
        print('first error: ', self.error[0])
        print('last error: ', self.error[-1])
        print('MSE: ', np.mean(self.error))

    def error_calc(self):
        pass

    def sigmoid(self, v, derivative=False):
        if derivative:
            return self.lmbda * v * (1 - v)
        return 1/(1+np.exp(-self.lmbda * v))

    def __str__(self):
        return str(self.__dict__)

    def save_model(self):
        pwd = os.path.dirname(os.path.abspath(__file__))
        np.savez_compressed(os.path.join(pwd+'\\saved_model', 'i2h_weights.npz'), self.i2h_weights)
        np.savez_compressed(os.path.join(pwd+'\\saved_model', 'h2o_weights.npz'), self.h2o_weights)

    def load_model(self):
        pwd = os.path.dirname(os.path.abspath(__file__))
        a = np.load(os.path.join(pwd+'\\saved_model', 'i2h_weights_working.npz'))['arr_0']
        b = np.load(os.path.join(pwd+'\\saved_model', 'h2o_weights_working.npz'))['arr_0']
        input_nodes = a.shape[1] - 1
        hidden_nodes = a.shape[0]
        output_nodes = b.shape[1]
        self.init_network(input_nodes, hidden_nodes, output_nodes)
        self.i2h_weights = a
        self.h2o_weights = b

    def normalize(self, input):
        return (input - self.min[:2])/(self.max[:2] - self.min[:2])

    def denormalize(self, output):
        return (output*(self.max[2:] - self.min[2:])) + self.min[2:]

    def matrix_multiplication(self, a, b):
        r = []
        for i in range(a.shape[0]):
            c = []
            for j in range(b.shape[1]):
                s = 0
                for k in range(b.shape[0]):
                    s += a[i][k] * b[k][j]
                c.append(s)
            r.append(c)
        return np.array(r)

import numpy as np
import os
np.set_printoptions(suppress=True)
np.random.seed(1)

class MyNN:
    def __init__(self):
        """This initializes all the variables required to smooth execution of the class.
        Do not forget to call **init_network** method with exactly 3 arguments after this.
        \nEx:
        obj = MyNN() \n
        obj.init_network(2,4,2)"""
        self.i2h_weights = None
        self.h2o_weights = None
        self.lmbda = 0.6
        self.alpha = 0.2
        self.eta = 0.4
        self.error = np.array([])
        self.validation_error = np.array([])
        self.output_delta_weight = 0.0
        self.hidden_delta_weight = 0.0
        self.max = None
        self.min = None

    def init_network(self, *args):
        '''This method initializes all the weights with random numbers between -1 and 1.
        The method will accept only 3 arguments:
        Input layer neurons = args[0]
        Hidden layer neurons = args[1]
        Output layer neurons = args[2] \n
        Raises ValueError otherwise
        '''
        if len(args) != 3:
            raise ValueError('Exactly 3 arguments required: input_nodes, hidden_nodes, output_nodes')
        input_nodes = args[0]
        hidden_nodes = args[1]
        output_nodes = args[2]
        self.i2h_weights = np.random.uniform(low=-1, high=1, size=(hidden_nodes, input_nodes+1))
        self.h2o_weights = np.random.uniform(low=-1, high=1, size=(output_nodes, hidden_nodes+1))


    def feed_forward(self, x, y, mode='train'):
        '''Internal method. Method for feed forward the network. This method is not meant to be called from outside
        of this class. Different modes you can pass are train, validate and predict. This method is
        called from the fit method internally.'''
        x = np.append(x, 1)
        # print('input: ', x)
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
        if mode == 'train':
            e = y.reshape(-1, 1) - output_av
            # self.error = np.append(self.error, np.square(e).mean())
            self.back_propagation(hidden_av, output_av, e, x)
            return np.square(e).mean()
        if mode == 'validate':
            e = y.reshape(-1, 1) - output_av
            # self.validation_error = np.append(self.validation_error, np.square(e).mean())
            return np.square(e).mean()
        if mode == 'predict':
            return output_av


    def output_gradient(self, v, e):
        '''Internal method. Calculates the local gradient for output layer neurons.'''
        return self.sigmoid(v, derivative=True) * e

    def hidden_gradient(self, h, grad_output):
        '''Internal method. Calculates the local gradient of hidden layer neuron.'''
        return self.sigmoid(h.reshape(-1, 1), derivative=True) * self.matrix_multiplication(self.h2o_weights.T, grad_output)

    def delta_weight_output(self, grad_output, hidden_av, prev_delta_weight):
        '''Internal method. Calculates delta weight for output layer neurons'''
        return self.eta * self.matrix_multiplication(grad_output, hidden_av.reshape(-1, 1).T) + self.alpha * prev_delta_weight

    def delta_weight_hidden(self, grad_hidden, x_arr, prev_delta_weight):
        '''Internal method. Calculates delta weight for the hidden layer neurons.'''
        return self.eta * self.matrix_multiplication(grad_hidden, x_arr.reshape(-1, 1).T) + self.alpha * prev_delta_weight

    def back_propagation(self, hidden_av, output_av, e, x):
        '''Internal method. This method is used to back-propagate and update the weights which were randomly initialized
        in the beginning.'''
        grad_output = self.output_gradient(output_av, e)
        grad_hidden = self.hidden_gradient(hidden_av, grad_output)
        self.output_delta_weight = self.delta_weight_output(grad_output, hidden_av, self.output_delta_weight)
        self.hidden_delta_weight = self.delta_weight_hidden(grad_hidden, x, self.hidden_delta_weight)
        self.h2o_weights = self.h2o_weights + self.output_delta_weight
        self.i2h_weights = self.i2h_weights + self.hidden_delta_weight[:-1, :]
        # Updating weight
        # print('old output wt: ', self.h2o_weights)
        # print('old hidden wt: ', self.i2h_weights)
        # print('new output wt: ', self.h2o_weights)
        # print('new hidden wt: ', self.i2h_weights)
        # print('error  :', e)
        # print('grad output: ', grad_output)
        # print('grad hidden: ', grad_hidden)
        # print('odw: ', self.output_delta_weight)
        # print('hdw: ', self.hidden_delta_weight)

    def  fit(self, inputs, outputs, val_in, val_out, epoch, min_delta=0.001, patience=10):
        """This method is for training the neural network. \n
        Parameters: \n
        \t inputs = training data inputs \n
        outputs = training data outputs
        val_in = validation data inputs
        val_out = validation data outputs
        min_delta = minimum change allowed to increase the validation mean squared error
        patience = minimum epochs to wait before stopping the training.
        """
        for t in range(epoch):
            e_epoch = 0.0
            val_e_epoch = 0.0
            for i, o in zip(inputs, outputs):
                e_row = self.feed_forward(i, o)
                e_epoch += e_row
            for i, o in zip(val_in, val_out):
                val_e_row = self.feed_forward(i, o, mode='validate')
                val_e_epoch += val_e_row
            self.validation_error = np.append(self.validation_error, np.sqrt(val_e_epoch))
            print('Epoch: {}, error: {}, rmse: {}, val_error: {}, val_rmse: {}'.format(t, e_epoch / len(inputs),
                        np.sqrt(e_epoch / len(inputs)),(val_e_epoch / len(val_in)),np.sqrt(val_e_epoch / len(val_in))))
            self.error = np.append(self.error, e_epoch/len(inputs))
            if len(self.validation_error) > patience and \
                    ((self.validation_error[-1] - self.validation_error[-10]) > min_delta):
                print('Training stopped to avoid over-fitting at epoch: ', t)
                print('Previous error was: ', self.error[-10])
                print('Current error is:', self.error[-1])
                break
        print('Printing final error-------------------------------')
        print('length of error: ', len(self.error))
        print('first error: ', self.error[0])
        print('last error: ', self.error[-1])
        print('MSE: ', np.mean(self.error))
        print('RMSE: ', np.sqrt(np.mean(self.error)))

    def validate(self, inputs, outputs):
        for i, o in zip(inputs, outputs):
            e_row = self.feed_forward(i, o, mode='validate')
            self.validation_error = np.append(self.validation_error, e_row)
        mse_validation = np.mean(self.validation_error)
        print('Printing final validation error-------------------------------')
        print('length of validation error: ', len(self.validation_error))
        print('MSE of validation error: ', mse_validation)
        print('RMSE of Validation error: ', np.sqrt(mse_validation))
        return mse_validation

    def predict(self, inputs):
        return self.feed_forward(inputs, None, mode='predict')

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
        np.savez_compressed(os.path.join(pwd + '\\saved_model', 'min_max_data.npz'), np.append(np.array(self.min), np.array(self.max)))

    def load_model(self):
        pwd = os.path.dirname(os.path.abspath(__file__))
        a = np.load(os.path.join(pwd+'\\saved_model', 'i2h_weights.npz'))['arr_0']
        b = np.load(os.path.join(pwd+'\\saved_model', 'h2o_weights.npz'))['arr_0']
        c = np.load(os.path.join(pwd+'\\saved_model', 'min_max_data.npz'))['arr_0']
        input_nodes = a.shape[1] - 1
        hidden_nodes = a.shape[0]
        output_nodes = b.shape[1]
        self.init_network(input_nodes, hidden_nodes, output_nodes)
        self.i2h_weights = a
        self.h2o_weights = b
        self.min = c[:4]
        self.max = c[4:]

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

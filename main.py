import numpy as np
from NeuralNetwork.mynn import MyNN
from NeuralNetwork.data_preprocessing import DataPreprocessing


if __name__ == '__main__':
    data = DataPreprocessing()
    data.load_data('C:\\Users\\kunal\\Documents\\PythonWork\\CE889_Assignment_1\\ce889_dataCollection.csv')
    x_train, x_test, y_train, y_test = data.preprocessing()
    print('max: ', data.max)
    print('min: ', data.min)
    a = MyNN()
    a.init_network(2,6,2)
    # print('Initialized weights: ', a)
    a.fit(x_train, y_train, 1000)
    a.save_model()
    # a.load_model()
    a.max = data.max
    a.min = data.min
    pred_arr = []
    err_arr = []
    for x, y in zip(x_test[:100], y_test[:100]):
        pred, err = a.feed_forward(x, y, predict=True)
        # pred_arr.append(pred)
        # err_arr.append(err)
        print('Input: ', x)
        print('Actual: ', y)
        print('Pred: ', pred.reshape(-1,))
        print('Denormalized prediction: ', a.denormalize(pred.reshape(-1, )))
        print('Denormalized actual: ', a.denormalize(y))
        print(' ')
    # print('Final weights: ', a)

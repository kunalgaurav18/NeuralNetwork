import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

class DataPreprocessing():
    def __init__(self):
        self.x_train = None
        self.x_test  = None
        self.y_train = None
        self.y_test = None
        self.df = None
        self.max = None
        self.min = None

    def load_data(self, filepath):
        self.df = pd.read_csv(filepath,header=None)

    def preprocessing(self):
        scaler = MinMaxScaler()
        scaler.fit(self.df)
        self.max = scaler.data_max_
        self.min = scaler.data_min_
        transformed_df = scaler.transform(self.df)
        x = transformed_df[:, :2]
        y = transformed_df[:, 2:]
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, test_size=0.2, random_state=30)
        return self.x_train, self.x_test, self.y_train, self.y_test

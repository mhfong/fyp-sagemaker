import pandas as pd
import numpy as np
import os
import yfinance as yf
from sklearn.model_selection import train_test_split

class Config:

    etf = "QQQ"

    dataset_type = "qqq"

    feature_columns = list(range(1,7))
    label_columns = [2,3]
    label_in_feature_index = (lambda x,y: [x.index(i) for i in y])(feature_columns, label_columns)

    predict_day = 1

    model_type = "gru"
    train_data_rate = 0.95

    do_train = True
    do_predict = True
    time_step = 20
    valid_data_rate = 0.15
    random_seed = 42
    shuffle_train_data = True

    input_size = len(feature_columns)
    output_size = len(label_columns)
    hidden_size = 128       
    layers = 2
    dropout_rate = 0.2
    batch_size = 64
    learning_rate = 0.001
    epoch = 100
    do_continue_train = False
    do_train_visualized = False
    patience = 5
    
    model_save_path = "./saved_model/"
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)

    data_save_path = "./dataset/"
    if not os.path.exists(data_save_path):
        os.makedirs(data_save_path)

class Data:
    def __init__(self, config):
        self.config = config
        self.data, self.data_column_name = self.read_data()

        self.data_num = self.data.shape[0]
        self.train_num = int(self.data_num * self.config.train_data_rate)

        # Normalization
        self.mean = np.mean(self.data, axis=0) 
        self.std = np.std(self.data, axis=0)
        self.norm_data = (self.data - self.mean)/self.std 

    def read_data(self):
        df = yf.download(self.config.etf)
        df.to_csv(self.config.data_save_path+self.config.etf+".csv")
        df = pd.read_csv(self.config.data_save_path+self.config.etf+".csv", usecols=self.config.feature_columns)
        return df.values, df.columns.tolist()

    def get_train_and_valid_data(self):
        feature_data = self.norm_data[:self.train_num]
        label_data = self.norm_data[self.config.predict_day : self.config.predict_day + self.train_num,
                                    self.config.label_in_feature_index]

        train_x = [feature_data[start_index + i*self.config.time_step : start_index + (i+1)*self.config.time_step]
                        for start_index in range(self.config.time_step)
                        for i in range((self.train_num - start_index) // self.config.time_step)]
        train_y = [label_data[start_index + i*self.config.time_step : start_index + (i+1)*self.config.time_step]
                    for start_index in range(self.config.time_step)
                    for i in range((self.train_num - start_index) // self.config.time_step)]

        train_x, train_y = np.array(train_x), np.array(train_y)

        train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, 
        test_size=self.config.valid_data_rate, random_state=self.config.random_seed, 
        shuffle=self.config.shuffle_train_data)
        
        return train_x, valid_x, train_y, valid_y

    def get_test_data(self):
        feature_data = self.norm_data[self.train_num:]
        sample_interval = min(feature_data.shape[0], self.config.time_step)
        self.start_num_in_test = feature_data.shape[0] % sample_interval
        time_step_size = feature_data.shape[0] // sample_interval

        test_x = [feature_data[self.start_num_in_test+i*sample_interval : self.start_num_in_test+(i+1)*sample_interval]
                   for i in range(time_step_size)]

        return np.array(test_x)
from torch.utils.data import Dataset
import numpy as np
import os
import re

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
pd.set_option('max_columns', 150)


class Dataset(Dataset):
    def __init__(self, root = './Data', charge_train=True, charge_test=False):

        building_meta_df = pd.read_csv(os.path.join(root, 'building_metadata.csv'))
        building_meta_df['primary_use'] = np.unique(building_meta_df.to_numpy()[:, 2], return_inverse=True)[1]
        self.building_meta_df = building_meta_df.to_numpy().astype(np.float32)

        if charge_train:
            train_df = pd.read_csv(os.path.join(root,'train.csv'))
            train_df["timestamp"] = pd.to_datetime(train_df["timestamp"])
            train_df['day'] = (((train_df['timestamp'] - train_df['timestamp'].min()) / np.timedelta64(1, 'D')) % 365)
            train_df['hour'] = (((train_df['timestamp'] - train_df['timestamp'].min()) / np.timedelta64(1, 'h')) % 24)
            del train_df['timestamp']

            weather_train_df = pd.read_csv(os.path.join(root, 'weather_train.csv'))
            weather_train_df["timestamp"] = pd.to_datetime(weather_train_df["timestamp"])
            weather_train_df['day'] = (((weather_train_df['timestamp'] - weather_train_df['timestamp'].min()) / np.timedelta64(1, 'D'))%365)
            weather_train_df['hour'] = (((weather_train_df['timestamp'] - weather_train_df['timestamp'].min()) / np.timedelta64(1, 'h'))%24)
            del weather_train_df['timestamp']

            self.train_df = train_df.to_numpy().astype(np.float32)
            self.weather_train_df = weather_train_df.to_numpy().astype(np.float32)
            self.train_charged = True

        if charge_test:
            test_df = pd.read_csv(os.path.join(root, 'test.csv'))
            test_df["timestamp"] = pd.to_datetime(train_df["timestamp"])
            test_df['day'] = (((test_df['timestamp'] - test_df['timestamp'].min()) / np.timedelta64(1, 'D')) % 365)
            test_df['hour'] = (((test_df['timestamp'] - test_df['timestamp'].min()) / np.timedelta64(1, 'h')) % 24)
            del test_df['timestamp']

            weather_test_df = pd.read_csv(os.path.join(root, 'weather_test.csv'))
            weather_test_df["timestamp"] = pd.to_datetime(weather_train_df["timestamp"])
            weather_test_df['day'] = (((weather_test_df['timestamp'] - weather_test_df['timestamp'].min()) / np.timedelta64(1, 'D')) % 365)
            weather_test_df['hour'] = (((weather_test_df['timestamp'] - weather_test_df['timestamp'].min()) / np.timedelta64(1, 'h')) % 24)
            del weather_test_df['timestamp']

            self.test_df = test_df.to_numpy().astype(np.float32)
            self.weather_test_df = weather_test_df.to_numpy().astype(np.float32)
            self.test_charged = True

        #Change Train or Test for charge one dataset or another
        self.charge = 'Train'

        to_print = 'Charged'
        if self.train_charged:
            to_print += ' Train Set '
        if self.test_charged:
            to_print += ' Test Set '
        print(to_print)

    def __len__(self):
        return len(self.train_df) if self.charge.lower() == 'train' else len(self.test_df)


    def __getitem__(self, idx):

        return {'input': None,
                'output': None}

Dataset()
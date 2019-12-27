from torch.utils.data import Dataset
import numpy as np
import os
import re

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
pd.set_option('max_columns', 150)

#DF indexes
METER_IDX = 1
Y_IDX = 2
#Building data Indexes
PRIMARY_USE_IDX = 2
#Lengths
METER_LEN = 4
DF_DATA_LEN = METER_LEN+2
TYPES_LEN = 16
BUILDING_DATA_LEN = TYPES_LEN+3
WEATHER_DATA_LEN = 10-3 #3 Keys
INPUT_LEN = DF_DATA_LEN+BUILDING_DATA_LEN+WEATHER_DATA_LEN

WEATHER_KEYS_IDX = [0, 8, 9]
WEATHER_VALUES_IDX = [1, 2, 3, 4, 5, 6, 7]

BUILDING_DATA_COLS_TO_STANDARIZE = ['square_feet', 'year_built', 'floor_count']
WEATHER_COLS_TO_STANDARIZE = ['air_temperature', 'cloud_coverage', 'dew_temperature', 'precip_depth_1_hr',
                              'sea_level_pressure', 'wind_direction', 'wind_speed']

class ASHRAEDataset(Dataset):
    def __init__(self, root = './Data', charge_train=True, charge_test=False, normalize=True):

        building_meta_df = pd.read_csv(os.path.join(root, 'building_metadata.csv'))
        building_meta_df['primary_use'] = np.unique(building_meta_df.to_numpy()[:, 2], return_inverse=True)[1]
        idx = [building_meta_df.columns.get_loc(col) for col in BUILDING_DATA_COLS_TO_STANDARIZE]
        self.building_meta_df = building_meta_df.to_numpy().astype(np.float32)
        self.building_meta_df = transform_nans(data=self.building_meta_df)
        self.building_meta_df = standarize(data=self.building_meta_df, idx=idx)
        if charge_train:
            train_df = pd.read_csv(os.path.join(root,'train.csv'))
            # We save the time as day and hour in separated fields, because hour and season are better descriptors
            train_df["timestamp"] = pd.to_datetime(train_df["timestamp"])
            train_df['day'] = np.round((((train_df['timestamp'] - train_df['timestamp'].min()) / np.timedelta64(1, 'D')) % 365),0)
            train_df['hour'] = np.round((((train_df['timestamp'] - train_df['timestamp'].min()) / np.timedelta64(1, 'h')) % 24),0)
            del train_df['timestamp']

            weather_train_df = pd.read_csv(os.path.join(root, 'weather_train.csv'))
            weather_train_df["timestamp"] = pd.to_datetime(weather_train_df["timestamp"])
            #We save the time as day and hour in separated fields, because hour and season are better descriptors
            weather_train_df['day'] = np.round((((weather_train_df['timestamp'] - weather_train_df['timestamp'].min()) / np.timedelta64(1, 'D'))%365),0)
            weather_train_df['hour'] = np.round((((weather_train_df['timestamp'] - weather_train_df['timestamp'].min()) / np.timedelta64(1, 'h'))%24),0)
            del weather_train_df['timestamp']
            idx = [weather_train_df.columns.get_loc(col) for col in WEATHER_COLS_TO_STANDARIZE]
            #Prepare weather data
            weather_train_df = weather_train_df.to_numpy()
            weather_train_df = transform_nans(data=weather_train_df)
            weather_train_df, weather_train_mean, weather_train_std = standarize(weather_train_df, idx=idx, return_mean_and_std=True)
            self.weather_train_df = {tuple(key) : value for key, value in zip(weather_train_df[:, WEATHER_KEYS_IDX].astype(np.int), weather_train_df[:, WEATHER_VALUES_IDX])}
            #Prepare df data
            self.train_df = train_df.to_numpy().astype(np.float32)
            self.train_df = transform_nans(data=self.train_df)

            self.train_charged = True

        if charge_test:
            test_df = pd.read_csv(os.path.join(root, 'test.csv'))
            # We save the time as day and hour in separated fields, because hour and season are better descriptors
            test_df["timestamp"] = pd.to_datetime(train_df["timestamp"])
            test_df['day'] = np.round((((test_df['timestamp'] - test_df['timestamp'].min()) / np.timedelta64(1, 'D')) % 365),0)
            test_df['hour'] = np.round((((test_df['timestamp'] - test_df['timestamp'].min()) / np.timedelta64(1, 'h')) % 24),0)
            del test_df['timestamp']
            # We save the time as day and hour in separated fields, because hour and season are better descriptors
            weather_test_df = pd.read_csv(os.path.join(root, 'weather_test.csv'))
            weather_test_df["timestamp"] = pd.to_datetime(weather_train_df["timestamp"])
            weather_test_df['day'] = np.round((((weather_test_df['timestamp'] - weather_test_df['timestamp'].min()) / np.timedelta64(1, 'D')) % 365),0)
            weather_test_df['hour'] = np.round((((weather_test_df['timestamp'] - weather_test_df['timestamp'].min()) / np.timedelta64(1, 'h')) % 24),0)
            del weather_test_df['timestamp']
            # Prepare weather data
            idx = [weather_test_df.columns.get_loc(col) for col in WEATHER_COLS_TO_STANDARIZE]
            weather_test_df = weather_test_df.to_numpy()
            weather_test_df = transform_nans(data=weather_test_df)
            weather_test_df = standarize(weather_test_df, idx=idx, mean=weather_train_mean, std=weather_train_std)
            self.weather_test_df = {tuple(key) : value for key, value in zip(weather_test_df[:, WEATHER_KEYS_IDX].astype(np.int), weather_test_df[:, WEATHER_VALUES_IDX])}
            # Prepare df data
            self.test_df = transform_nans(data=self.test_df)
            self.test_df = test_df.to_numpy().astype(np.float32)

            self.test_charged = True

        #Change Train or Test for charge one dataset or another
        self.charge = 'Train'

        to_print = 'Charged'
        if charge_train:
            to_print += ' Train Set '
        if charge_test:
            to_print += ' Test Set '
        print(to_print)

    def __len__(self):
        return len(self.train_df) if self.charge.lower() == 'train' else len(self.test_df)


    def __getitem__(self, idx):
        building_meta = self.building_meta_df
        if self.charge.lower() == 'train':
            df = self.train_df
            weather = self.weather_train_df
        else:
            df = self.test_df
            weather = self.weather_test_df
        df_data = df[idx]
        building_data = building_meta[int(df_data[0])]
        weather_keys = tuple(np.round([int(building_data[0]), int(df_data[-2]), int(df_data[-1])],3))
        if weather_keys not in weather:
            weather_keys = get_nearest_day_info(weather_keys, weather)
        weather_data = weather[weather_keys]
        x = construct_x(df_data, building_data, weather_data)
        y = df_data[Y_IDX]
        return(x,y)

def construct_x(df_data, building_data, weather_data):
    x = np.zeros(shape=INPUT_LEN, dtype=np.float32)
    #One Hot Encoding the Meter Variable
    x[int(df_data[METER_IDX])] = 1.
    #Including Day and Hour of the measurement
    x[METER_LEN:DF_DATA_LEN] = df_data[-2:]
    #One Hot Encoding the Primary Use Variable
    x[DF_DATA_LEN+int(building_data[PRIMARY_USE_IDX])] = 1.
    #Including Square Feet, Year Built and Floor Count
    x[DF_DATA_LEN+TYPES_LEN:DF_DATA_LEN+BUILDING_DATA_LEN] = building_data[-3:]
    #Including Air_temperature, Cloud_Coverage, Dew_temperature, Precip Depth 1 hour,
    #Sea Level Presure, Wind Direction and Wind Speed
    x[DF_DATA_LEN+BUILDING_DATA_LEN:] = weather_data
    return x

def standarize(data, idx, mean=None, std=None, return_mean_and_std=False):
    if mean is None:
        mean = np.mean(data[...,idx],axis=0)
    if std is None:
        std = np.std(data[...,idx],axis=0)
    data[...,idx] = (data[...,idx]-mean)/std
    if not return_mean_and_std:
        return data
    else:
        return data, mean, std

def transform_nans(data, operation='mean'):
    for i in range(data.shape[-1]):
        if operation.lower() == 'mean':
            data[np.isnan(data[...,i]),i] = np.nanmean(data[...,i])
        elif operation.lower() == 'median':
            data[np.isnan(data[..., i]), i] = np.nanmedian(data[..., i])
    return data

def get_nearest_day_info(key, data):
    site_id, day, hour = key
    while(not (site_id, day, hour) in data):
        day = (day+1)%365
    return (site_id, day, hour)
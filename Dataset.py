from torch.utils.data import Dataset
import numpy as np
import os
import re

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
pd.set_option('max_columns', 150)

import random, math, psutil, pickle

class Dataset(Dataset):
    def __init__(self, root = './Data', charge_train=True, charge_test=False):
        train_df = pd.read_csv(os.path.join(root,'train.csv'))
        train_df["timestamp"] = pd.to_datetime(train_df["timestamp"], format='%Y-%m-%d %H:%M:%S')
        print('Charged')
        #self.dirs = self.get_all_files(original_dir=os.path.join(root_dir))


    def get_all_files(self, original_dir, cum_dir='', key_word_to_introduce=[''], key_word_to_discard=['####']):
        files = []
        for actual in os.listdir(os.path.join(original_dir, cum_dir)):
            path = os.path.join(original_dir, cum_dir, actual)
            if os.path.isfile(path):
                    file_dir = os.path.join(cum_dir, actual)
                    has_key = len([k for k in key_word_to_introduce if k in file_dir]) == len(key_word_to_introduce)
                    discardable = len([k for k in key_word_to_discard if k in file_dir])>0
                    if has_key and not discardable:
                        files.extend([file_dir[:file_dir.rfind('.')]])
            else:
                files.extend(self.get_all_files(original_dir=original_dir, cum_dir=os.path.join(cum_dir, actual),
                                                key_word_to_introduce=key_word_to_introduce,
                                                key_word_to_discard=key_word_to_discard))
        return files

    def __len__(self):
        return len(self.dirs)


    def __getitem__(self, idx):

        return {'input': None,
                'output': None}

Dataset()
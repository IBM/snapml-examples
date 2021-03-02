# Copyright 2021 IBM Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from .dataset import Dataset
import os
import io
import subprocess
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler, normalize
from zipfile import ZipFile

class CreditCardFraud(Dataset):

    def __init__(self, cache_dir):
        files = ['creditcard.X_train.npy',
                 'creditcard.X_test.npy',
                 'creditcard.y_train.npy',
                 'creditcard.y_test.npy']
        super().__init__(cache_dir, type(self).__name__, files)
        self.raw_file = os.path.join(self.working_dir, 'creditcardfraud.zip')
     
    def download_raw_data(self):

        p = subprocess.Popen(['kaggle', 'datasets', 'download', 'mlg-ulb/creditcardfraud'], cwd=self.working_dir)
        p.wait()

        if not os.path.isfile(self.raw_file):
            raise RuntimeError(
                "Could not download dataset from Kaggle. Please ensure you have installed a Kaggle API token: https://www.kaggle.com/docs/api"
            ) 

    def preprocess_data(self):

        with ZipFile(self.raw_file, 'r') as a:
            df = pd.read_csv(io.BytesIO(a.read('creditcard.csv')))
        
        df.iloc[:, 1:29] = StandardScaler().fit_transform(df.iloc[:, 1:29])

        data_matrix = df.values
        X = data_matrix[:, 1:29]
        y = data_matrix[:, 30]

        # Normalize the data
        X = normalize(X, norm="l1")

        stratSplit = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=42)

        for train_index, test_index in stratSplit.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

        return X_train, X_test, y_train, y_test

    def write_cache_data(self, X_train, X_test, y_train, y_test):
        np.save(os.path.join(self.working_dir, 'creditcard.X_train'), X_train)
        np.save(os.path.join(self.working_dir, 'creditcard.X_test'), X_test)
        np.save(os.path.join(self.working_dir, 'creditcard.y_train'), y_train)
        np.save(os.path.join(self.working_dir, 'creditcard.y_test'), y_test)

    def read_cache_data(self):
        X_train = np.load(os.path.join(self.working_dir, 'creditcard.X_train.npy'))
        X_test = np.load(os.path.join(self.working_dir, 'creditcard.X_test.npy'))
        y_train = np.load(os.path.join(self.working_dir, 'creditcard.y_train.npy'))
        y_test = np.load(os.path.join(self.working_dir, 'creditcard.y_test.npy'))
        return X_train, X_test, y_train, y_test

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
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize

class Higgs(Dataset):

    def __init__(self, cache_dir):
        files = ['HIGGS.X_train.npy',
                 'HIGGS.X_test.npy',
                 'HIGGS.y_train.npy',
                 'HIGGS.y_test.npy']
        super().__init__(cache_dir, type(self).__name__, files)
        self.raw_file = os.path.join(self.working_dir, 'HIGGS.csv.gz')
     
    def download_raw_data(self):
        self._download_file('https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz', self.raw_file)

    def preprocess_data(self):
        df = pd.read_csv(self.raw_file, compression='gzip', header=None)
        y = df.pop(0).values
        X = df.values
        X = normalize(X, axis=1, norm='l1')
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
        return X_train, X_test, y_train, y_test

    def write_cache_data(self, X_train, X_test, y_train, y_test):
        np.save(os.path.join(self.working_dir, 'HIGGS.X_train'), X_train)
        np.save(os.path.join(self.working_dir, 'HIGGS.X_test'), X_test)
        np.save(os.path.join(self.working_dir, 'HIGGS.y_train'), y_train)
        np.save(os.path.join(self.working_dir, 'HIGGS.y_test'), y_test)

    def read_cache_data(self):
        X_train = np.load(os.path.join(self.working_dir, 'HIGGS.X_train.npy'))
        X_test = np.load(os.path.join(self.working_dir, 'HIGGS.X_test.npy'))
        y_train = np.load(os.path.join(self.working_dir, 'HIGGS.y_train.npy'))
        y_test = np.load(os.path.join(self.working_dir, 'HIGGS.y_test.npy'))
        return X_train, X_test, y_train, y_test


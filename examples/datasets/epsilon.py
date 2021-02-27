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
import subprocess
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split
import numpy as np

class Epsilon(Dataset):

    def __init__(self, cache_dir):
        files = ['epsilon.X_train.npy',
                 'epsilon.X_test.npy',
                 'epsilon.y_train.npy',
                 'epsilon.y_test.npy']
        super().__init__(cache_dir, type(self).__name__, files)
        self.raw_file = os.path.join(self.working_dir, 'epsilon_normalized.bz2')
     
    def download_raw_data(self):
        self._download_file('https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/epsilon_normalized.bz2', self.raw_file)

    def preprocess_data(self):
        X, y = load_svmlight_file(self.raw_file)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
        X_train = np.array(X_train.todense(), dtype=np.float32)
        X_test = np.array(X_test.todense(), dtype=np.float32)
        return X_train, X_test, y_train, y_test

    def write_cache_data(self, X_train, X_test, y_train, y_test):
        np.save(os.path.join(self.working_dir, 'epsilon.X_train'), X_train)
        np.save(os.path.join(self.working_dir, 'epsilon.X_test'), X_test)
        np.save(os.path.join(self.working_dir, 'epsilon.y_train'), y_train)
        np.save(os.path.join(self.working_dir, 'epsilon.y_test'), y_test)

    def read_cache_data(self):
        X_train = np.load(os.path.join(self.working_dir, 'epsilon.X_train.npy'))
        X_test = np.load(os.path.join(self.working_dir, 'epsilon.X_test.npy'))
        y_train = np.load(os.path.join(self.working_dir, 'epsilon.y_train.npy'))
        y_test = np.load(os.path.join(self.working_dir, 'epsilon.y_test.npy'))
        return X_train, X_test, y_train, y_test


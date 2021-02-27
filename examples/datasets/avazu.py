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
from sklearn.datasets import load_svmlight_file
from scipy.sparse import save_npz, load_npz
import numpy as np

class Avazu(Dataset):

    def __init__(self, cache_dir):
        files = ['avazu.X_train.npz', 
                 'avazu.X_test.npz',
                 'avazu.y_train.npy',
                 'avazu.y_test.npy']
        super().__init__(cache_dir, type(self).__name__, files)
        self.raw_train = os.path.join(self.working_dir, 'avazu-app.tr.bz2')
        self.raw_test = os.path.join(self.working_dir, 'avazu-app.val.bz2')

    def download_raw_data(self):
        self._download_file('https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/avazu-app.tr.bz2', self.raw_train)
        self._download_file('https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/avazu-app.val.bz2', self.raw_test)

    def preprocess_data(self):
        X_train, y_train = load_svmlight_file(self.raw_train, n_features=1_000_000)
        X_test, y_test = load_svmlight_file(self.raw_test, n_features=1_000_000)
        return X_train, X_test, y_train, y_test

    def write_cache_data(self, X_train, X_test, y_train, y_test):
        save_npz(os.path.join(self.working_dir, 'avazu.X_train'), X_train, compressed=False)
        save_npz(os.path.join(self.working_dir, 'avazu.X_test'), X_test,  compressed=False)
        np.save(os.path.join(self.working_dir, 'avazu.y_train'), y_train)
        np.save(os.path.join(self.working_dir, 'avazu.y_test'), y_test)

    def read_cache_data(self):
        X_train = load_npz(os.path.join(self.working_dir, 'avazu.X_train.npz'))
        X_test  = load_npz(os.path.join(self.working_dir, 'avazu.X_test.npz'))
        y_train = np.load(os.path.join(self.working_dir, 'avazu.y_train.npy'))
        y_test  = np.load(os.path.join(self.working_dir, 'avazu.y_test.npy'))
        return X_train, X_test, y_train, y_test


           

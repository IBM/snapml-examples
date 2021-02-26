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

import os
import subprocess
import numpy as np
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from scipy.sparse import save_npz, load_npz

class Mnist8m():

    def __init__(self, cache_dir):
        self.name = type(self).__name__
        self.cache_dir = cache_dir
        self.working_dir = os.path.join(self.cache_dir, self.name)
        self.files = ['mnist8m.X_train.npz',
                      'mnist8m.X_test.npz',
                      'mnist8m.y_train.npy',
                      'mnist8m.y_test.npy']

    def __check_files_exist(self, files):
        files_exist = True
        for file in files:
            files_exist &= os.path.isfile(os.path.join(self.working_dir, file))
        return files_exist

    def get_train_test_split(self):

        if not self.__check_files_exist(self.files):

            print("files do not exist")

            p = subprocess.Popen(['mkdir', '-p', self.working_dir])
            p.wait()

            p = subprocess.Popen(['wget', 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/mnist8m.scale.bz2'], cwd=self.working_dir)
            p.wait()

            p = subprocess.Popen(['bunzip2', 'mnist8m.scale.bz2'], cwd=self.working_dir)
            p.wait()

            X, y = load_svmlight_file(os.path.join(self.working_dir, "mnist8m.scale"))

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.25, random_state=42
            )

            X_train = normalize(X_train, axis=1, norm="l1")
            X_test = normalize(X_test, axis=1, norm="l1")

  
            save_npz(os.path.join(self.working_dir, 'mnist8m.X_train'), X_train, compressed=False)
            save_npz(os.path.join(self.working_dir, 'mnist8m.X_test'), X_test,  compressed=False)
            np.save(os.path.join(self.working_dir, 'mnist8m.y_train'), y_train)
            np.save(os.path.join(self.working_dir, 'mnist8m.y_test'), y_test)

        else:

            X_train = load_npz(os.path.join(self.working_dir, 'mnist8m.X_train.npz'))
            X_test  = load_npz(os.path.join(self.working_dir, 'mnist8m.X_test.npz'))
            y_train = np.load(os.path.join(self.working_dir, 'mnist8m.y_train.npy'))
            y_test = np.load(os.path.join(self.working_dir, 'mnist8m.y_test.npy'))

        return X_train, X_test, y_train, y_test
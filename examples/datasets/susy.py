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

class Susy():

    def __init__(self, cache_dir):
        self.name = type(self).__name__
        self.cache_dir = cache_dir
        self.working_dir = os.path.join(self.cache_dir, self.name)
        self.files = ['SUSY.X_train.npy',
                      'SUSY.X_test.npy',
                      'SUSY.y_train.npy',
                      'SUSY.y_test.npy']

    def __check_files_exist(self, files):
        files_exist = True
        for file in files:
            files_exist &= os.path.isfile(os.path.join(self.working_dir, file))
        return files_exist

    def get_train_test_split(self):

        if not self.__check_files_exist(self.files):

            p = subprocess.Popen(['mkdir', '-p', self.working_dir])
            p.wait()

            p = subprocess.Popen(['wget', 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/SUSY.bz2'], cwd=self.working_dir)
            p.wait()

            p = subprocess.Popen(['bunzip2', 'SUSY.bz2'], cwd=self.working_dir)
            p.wait()

            X, y = load_svmlight_file(os.path.join(self.working_dir, "SUSY"))

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.25, random_state=42
            )

            X_train = np.array(X_train.todense(), dtype=np.float32)
            X_test = np.array(X_test.todense(), dtype=np.float32)

            X_train = normalize(X_train, axis=1, norm="l1")
            X_test = normalize(X_test, axis=1, norm="l1")

            np.save(os.path.join(self.working_dir, 'SUSY.X_train'), X_train)
            np.save(os.path.join(self.working_dir, 'SUSY.X_test'), X_test)
            np.save(os.path.join(self.working_dir, 'SUSY.y_train'), y_train)
            np.save(os.path.join(self.working_dir, 'SUSY.y_test'), y_test)

        else:

            X_train = np.load(os.path.join(self.working_dir, 'SUSY.X_train.npy'))
            X_test = np.load(os.path.join(self.working_dir, 'SUSY.X_test.npy'))
            y_train = np.load(os.path.join(self.working_dir, 'SUSY.y_train.npy'))
            y_test = np.load(os.path.join(self.working_dir, 'SUSY.y_test.npy'))

        return X_train, X_test, y_train, y_test


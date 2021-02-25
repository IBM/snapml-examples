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
from scipy.sparse import save_npz, load_npz
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, OneHotEncoder, normalize
from scipy.sparse import save_npz, load_npz, csr_matrix

class Allstate():

    def __init__(self, cache_dir):
        self.name = type(self).__name__
        self.cache_dir = cache_dir
        self.working_dir = os.path.join(self.cache_dir, self.name)
        self.files = ['allstate.X_train.npy', 
                      'allstate.y_train.npy', 
                      'allstate.X_test.npy', 
                      'allstate.y_test.npy']

    def __check_files_exist(self, files):
        files_exist = True
        for file in files:
            files_exist &= os.path.isfile(os.path.join(self.working_dir, file))
        return files_exist

    def get_train_test_split(self):

        if not self.__check_files_exist(self.files):


            p = subprocess.Popen(['mkdir', '-p', self.working_dir])
            p.wait()

            p = subprocess.Popen(['kaggle', 'competitions', 'download', '-c', 'ClaimPredictionChallenge'], cwd=self.working_dir)
            p.wait()

            p = subprocess.Popen(['unzip', 'ClaimPredictionChallenge.zip'], cwd=self.working_dir)
            p.wait()
            
            p = subprocess.Popen(['unzip', 'train_set.zip'], cwd=self.working_dir)
            p.wait()
            

            df = pd.read_csv(os.path.join(self.working_dir, 'train_set.csv'))
            df.drop(['Row_ID'],axis=1,inplace=True)
            df.replace('?', np.NaN, inplace=True)
            df.fillna(-1, inplace=True)

            df_Y = df[['Claim_Amount']]
            df_X = df.drop(['Claim_Amount'],axis=1)

            indices = range(df_X.shape[0])
            X_train, X_test, y_train, y_test  = train_test_split(df_X, df_Y, test_size=0.3, shuffle=True, random_state=42)
        
            X_train = X_train.reset_index().copy()
            X_test = X_test.reset_index().copy()
         
            
            for col in ['Blind_Make', 'Blind_Model', 'Blind_Submodel', 'NVCat']:
                X_train[col] = X_train[col].astype(str)
                X_test[col] = X_test[col].astype(str)
                
                le = LabelEncoder()

                # fit label encoder
                le.fit(X_train[col])

                # extract dictionary mapping from label encoder
                le_dict = dict(zip(le.classes_, le.transform(le.classes_)))

                # replace categories with labels (assign label -1 if not in dict)
                X_train[col] = X_train[col].map(lambda x: le_dict.get(x, -1))
                X_test[col] = X_test[col].map(lambda x: le_dict.get(x, -1))

            
            categoricalColumns = ['Cat1', 'Cat2', 'Cat3', 'Cat4', 'Cat5', 'Cat6', 'Cat7', 'Cat8', 'Cat9', 'Cat10', 'Cat11', 'Cat12']
            for col in categoricalColumns:
                X_train[col] = X_train[col].astype(str)
                X_test[col] = X_test[col].astype(str)

                oh = OneHotEncoder(handle_unknown='ignore')
                tmp_train = oh.fit_transform(X_train[col].values.reshape(-1,1)).toarray()
                tmp_test = oh.transform(X_test[col].values.reshape(-1,1)).toarray()

                assert(tmp_train.shape[1] == tmp_test.shape[1])

                X_train = X_train.drop(col,axis=1)
                X_test = X_test.drop(col,axis=1)

                new_cols = [col + '-' + str(int(i)) for i in range(tmp_train.shape[1])]

                tmp_train = pd.DataFrame(tmp_train, columns=new_cols)
                tmp_test  = pd.DataFrame(tmp_test,  columns=new_cols)

                X_train = pd.concat([X_train, tmp_train],axis=1)
                X_test  = pd.concat([X_test,  tmp_test],axis=1)

            for col in ['Calendar_Year']:
                X_train['Vehicle_Age'] = X_train['Calendar_Year'] - X_train['Model_Year']
                X_test['Vehicle_Age'] = X_test['Calendar_Year'] - X_test['Model_Year']

            X_train.drop(['Calendar_Year'], axis=1, inplace=True)
            X_test.drop(['Calendar_Year'], axis=1, inplace=True)

            X_train = np.ascontiguousarray(X_train.values, dtype=np.float32)
            X_test = np.ascontiguousarray(X_test.values, dtype=np.float32)

            min_max_scaler = MinMaxScaler()
            min_max_scaler.fit(X_train)
            X_train = min_max_scaler.transform(X_train)
            X_test = min_max_scaler.transform(X_test)

            # normalize and cast array to float32
            X_train = normalize(X_train, axis=1, norm='l1')
            X_test = normalize(X_test, axis=1, norm='l1')

            y_train = y_train.values.ravel()
            y_test = y_test.values.ravel()

            y_train = (y_train > 0).astype(np.float32)
            y_test = (y_test > 0).astype(np.float32)

            np.save(os.path.join(self.working_dir, 'allstate.X_train'), y_train)
            np.save(os.path.join(self.working_dir, 'allstate.X_test'), y_test)
            np.save(os.path.join(self.working_dir, 'allstate.y_train'), y_train)
            np.save(os.path.join(self.working_dir, 'allstate.y_test'), y_test)
            

        else:

            X_train = np.load(os.path.join(self.working_dir, 'allstate.X_train.npy'))
            X_test = np.load(os.path.join(self.working_dir, 'allstate.X_test.npy'))
            y_train = np.load(os.path.join(self.working_dir, 'allstate.y_train.npy'))
            y_test = np.load(os.path.join(self.working_dir, 'allstate.y_test.npy'))
      
        return X_train, X_test, y_train, y_test
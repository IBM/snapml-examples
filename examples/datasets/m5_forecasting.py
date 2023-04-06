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
from sklearn.preprocessing import LabelEncoder
from zipfile import ZipFile

class M5Forecasting(Dataset):

    def __init__(self, cache_dir):
        files = ['m5forecasting.X_train.npy',
                 'm5forecasting.X_test.npy',
                 'm5forecasting.y_train.npy',
                 'm5forecasting.y_test.npy']
        super().__init__(cache_dir, type(self).__name__, files)
        self.raw_file = os.path.join(self.working_dir, 'm5-forecasting-accuracy.zip')

    def download_raw_data(self):

        p = subprocess.Popen(['kaggle', 'competitions', 'download', '-c', 'm5-forecasting-accuracy'], cwd=self.working_dir)
        p.wait()

        if not os.path.isfile(self.raw_file):
            raise RuntimeError(
                "Could not download dataset from Kaggle. Please ensure you have installed a Kaggle API token: https://www.kaggle.com/docs/api"
            )

    def preprocess_data(self):

        # Data for the last year
        firstDay = 1535
        lastDay = 1900

        # Use 365 sales days (columns) for training
        numCols = [f"d_{day}" for day in range(firstDay, lastDay+1)]

        # Define all categorical columns
        catCols = ['id', 'item_id', 'dept_id','store_id', 'cat_id', 'state_id']

        # Read files
        with ZipFile(self.raw_file, 'r') as a:
             calendar = pd.read_csv(io.BytesIO(a.read('calendar.csv')))
             prices = pd.read_csv(io.BytesIO(a.read('sell_prices.csv')))
             df = pd.read_csv(io.BytesIO(a.read('sales_train_validation.csv')), usecols = catCols + numCols)

        calendar["date"] = pd.to_datetime(calendar["date"])

        df = pd.melt(df,
                      id_vars = catCols,
                      value_vars = [col for col in df.columns if col.startswith("d_")],
                      var_name = "d",
                      value_name = "sales")

        # Merge "df" with "calendar" and "prices" dataframe
        df = df.merge(calendar, on = "d", copy = False)
        df = df.merge(prices, on = ["store_id", "item_id", "wm_yr_wk"], copy = False)

        # Lag features for 1 week, 1 month period
        dayLags = [7, 28]
        lagSalesCols = [f"lag_{dayLag}" for dayLag in dayLags]
        for dayLag, lagSalesCol in zip(dayLags, lagSalesCols):
            df[lagSalesCol] = df[["id","sales"]].groupby("id")["sales"].shift(dayLag).fillna(-1)

        # Rolling mean features for 1 week, 1 month period
        windows = [7, 28]
        for window in windows:
            for dayLag, lagSalesCol in zip(dayLags, lagSalesCols):
                df[f"rmean_{dayLag}_{window}"] = df[["id", lagSalesCol]].groupby("id")[lagSalesCol].transform(lambda x: x.rolling(window).mean()).fillna(-1)


        # Test dataset -> Last 28 days
        cutoff = df.date.max() - pd.to_timedelta(28, unit = 'D')
        xtrain = df.loc[df.date < cutoff].copy()
        xtest = df.loc[df.date >= cutoff].copy()


        xtrain.drop(['id', 'd', 'wm_yr_wk','snap_WI', 'snap_CA', 'snap_TX', 'date', 'weekday', 'month', 'year', 'event_name_1', 'event_type_1', 'event_name_2', 'event_type_2'], axis = 1, inplace = True)
        xtrain.dropna()

        xtest.drop(['id', 'd', 'wm_yr_wk', 'snap_WI', 'snap_CA', 'snap_TX', 'date', 'weekday', 'month', 'year',  'event_name_1', 'event_type_1', 'event_name_2', 'event_type_2'], axis = 1, inplace = True)
        xtest.dropna()

        # encode categorical features
        cat_feats = ['item_id', 'dept_id','store_id', 'cat_id', 'state_id'  ]
        for cf in cat_feats:
            enc = LabelEncoder()
            xtrain[cf] = enc.fit_transform(xtrain[cf])
            xtest[cf] = enc.transform(xtest[cf])


        y_train = xtrain["sales"].to_numpy()
        X_train = xtrain.drop(['sales'], axis = 1).to_numpy()
        y_test = xtest["sales"].to_numpy()
        X_test = xtest.drop(['sales'], axis = 1).to_numpy()

        return X_train, X_test, y_train, y_test

    def write_cache_data(self, X_train, X_test, y_train, y_test):
        np.save(os.path.join(self.working_dir, 'm5forecasting.X_train'), X_train)
        np.save(os.path.join(self.working_dir, 'm5forecasting.X_test'), X_test)
        np.save(os.path.join(self.working_dir, 'm5forecasting.y_train'), y_train)
        np.save(os.path.join(self.working_dir, 'm5forecasting.y_test'), y_test)

    def read_cache_data(self):
        X_train = np.load(os.path.join(self.working_dir, 'm5forecasting.X_train.npy'))
        X_test = np.load(os.path.join(self.working_dir, 'm5forecasting.X_test.npy'))
        y_train = np.load(os.path.join(self.working_dir, 'm5forecasting.y_train.npy'))
        y_test = np.load(os.path.join(self.working_dir, 'm5forecasting.y_test.npy'))
        return X_train, X_test, y_train, y_test


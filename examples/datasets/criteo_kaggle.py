import os
import subprocess
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split
import numpy as np
from scipy.sparse import load_npz, save_npz

class CriteoKaggle():

    def __init__(self, cache_dir):
        self.name = type(self).__name__
        self.cache_dir = cache_dir
        self.working_dir = os.path.join(self.cache_dir, self.name)
        self.files = ['criteo.kaggle2014.X_train.npz', 
                      'criteo.kaggle2014.X_test.npz',
                      'criteo.kaggle2014.y_train.npy',
                      'criteo.kaggle2014.y_test.npy']

    def __check_files_exist(self, files):
        files_exist = True
        for file in files:
            files_exist &= os.path.isfile(os.path.join(self.working_dir, file))
        return files_exist

    def get_train_test_split(self):

        if not self.__check_files_exist(self.files):

            p = subprocess.Popen(['mkdir', '-p', self.working_dir])
            p.wait()

            p = subprocess.Popen(['wget', 'https://s3-us-west-2.amazonaws.com/criteo-public-svm-data/criteo.kaggle2014.svm.tar.gz'], cwd=self.working_dir)
            p.wait()

            p = subprocess.Popen(['tar', '-xzf', 'criteo.kaggle2014.svm.tar.gz'], cwd=self.working_dir)
            p.wait()

            X, y = load_svmlight_file(os.path.join(self.working_dir,'criteo.kaggle2014.train.svm'))

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

            save_npz(os.path.join(self.working_dir, 'criteo.kaggle2014.X_train'), X_train, compressed=False)
            save_npz(os.path.join(self.working_dir, 'criteo.kaggle2014.X_test'), X_test,  compressed=False)

            np.save(os.path.join(self.working_dir, 'criteo.kaggle2014.y_train'), y_train)
            np.save(os.path.join(self.working_dir, 'criteo.kaggle2014.y_test'), y_test)

        else:

            X_train = load_npz(os.path.join(self.working_dir, 'criteo.kaggle2014.X_train.npz'))
            X_test  = load_npz(os.path.join(self.working_dir, 'criteo.kaggle2014.X_test.npz'))
            y_train = np.load(os.path.join(self.working_dir, 'criteo.kaggle2014.y_train.npy'))
            y_test  = np.load(os.path.join(self.working_dir, 'criteo.kaggle2014.y_test.npy'))

        return X_train, X_test, y_train, y_test


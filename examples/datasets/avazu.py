import os
import subprocess
from sklearn.datasets import load_svmlight_file
from scipy.sparse import save_npz, load_npz
import numpy as np

class Avazu():

    def __init__(self, cache_dir):
        self.name = type(self).__name__
        self.cache_dir = cache_dir
        self.working_dir = os.path.join(self.cache_dir, self.name)
        self.files = ['avazu.X_train.npz', 
                      'avazu.X_test.npz',
                      'avazu.y_train.npy',
                      'avazu.y_test.npy']

    def __check_files_exist(self, files):
        files_exist = True
        for file in files:
            files_exist &= os.path.isfile(os.path.join(self.working_dir, file))
        return files_exist

    def get_train_test_split(self):

        if not self.__check_files_exist(self.files):

            p = subprocess.Popen(['mkdir', '-p', self.working_dir])
            p.wait()

            for name in ['avazu-app.tr', 'avazu-app.val']:
                p = subprocess.Popen(['wget', 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/%s.bz2' % (name)], cwd=self.working_dir)
                p.wait()

                p = subprocess.Popen(['bunzip2', '%s.bz2' % (name)], cwd=self.working_dir)
                p.wait()


            X_train, y_train = load_svmlight_file(os.path.join(self.working_dir, 'avazu-app.tr'), n_features=1_000_000)
            X_test, y_test = load_svmlight_file(os.path.join(self.working_dir, 'avazu-app.val'), n_features=1_000_000)

            save_npz(os.path.join(self.working_dir, 'avazu.X_train'), X_train, compressed=False)
            save_npz(os.path.join(self.working_dir, 'avazu.X_test'), X_test,  compressed=False)

            np.save(os.path.join(self.working_dir, 'avazu.y_train'), y_train)
            np.save(os.path.join(self.working_dir, 'avazu.y_test'), y_test)

        else:

            X_train = load_npz(os.path.join(self.working_dir, 'avazu.X_train.npz'))
            X_test  = load_npz(os.path.join(self.working_dir, 'avazu.X_test.npz'))
            y_train = np.load(os.path.join(self.working_dir, 'avazu.y_train.npy'))
            y_test  = np.load(os.path.join(self.working_dir, 'avazu.y_test.npy'))
      
        return X_train, X_test, y_train, y_test

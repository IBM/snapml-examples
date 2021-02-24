import os
import subprocess
from sklearn.datasets import load_svmlight_file

class Avazu():

    def __init__(self, cache_dir):
        self.cache_dir = cache_dir
        self.working_dir = os.path.join(self.cache_dir, type(self).__name__)
        self.files = ['avazu-app.tr', 'avazu-app.val']

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

        return X_train, X_test, y_train, y_test

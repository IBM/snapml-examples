import os
import requests
from tqdm.notebook import tqdm

class Dataset():

    def __init__(self, cache_dir, name, files):
        self.name = name
        self.cache_dir = cache_dir
        self.working_dir = os.path.join(self.cache_dir, self.name)
        self.files = files

    def __check_cache_exist(self):
        files_exist = True
        for file in self.files:
            files_exist &= os.path.isfile(os.path.join(self.working_dir, file))
        return files_exist

    def _download_file(self, url, filename):
        print("Downloading file: %s" % (url))
        r = requests.get(url, allow_redirects=True, stream=True)
        tot_size = int(r.headers.get('content-length', 0))
        if os.path.isfile(filename) and os.stat(filename).st_size == tot_size:
            print("File %s with correct size exists; skipping download" % (filename))
            return
        pb = tqdm(total=tot_size, unit='iB', unit_scale=True)
        with open(filename, 'wb') as f:
            for data in r.iter_content(1024):
                pb.update(len(data))
                f.write(data)
        pb.close()

    def download_raw_data(self):
        pass

    def preprocess_data(self):
        pass

    def write_cache_data(self, X_train, X_test, y_train, y_test):
        pass

    def read_cache_data(self):
        pass

    def get_train_test_split(self):

        if self.__check_cache_exist():
            print("Reading binary %s dataset (cache) from disk." % (self.name))
            return self.read_cache_data()

        print("Creating working directory: %s" % (self.working_dir))
        os.makedirs(self.working_dir, exist_ok=True)

        print("Downloading %s dataset." % (self.name))
        print("Please note: subsequent calls to `get_train_test_split` will read cached binary data, and thus be much faster.")
        self.download_raw_data()

        print("Preprocessing %s dataset." % (self.name))
        X_train, X_test, y_train, y_test = self.preprocess_data()

        print("Writing binary %s dataset (cache) to disk." % (self.name))
        self.write_cache_data(X_train, X_test, y_train, y_test)

        assert self.__check_cache_exist()

        return X_train, X_test, y_train, y_test


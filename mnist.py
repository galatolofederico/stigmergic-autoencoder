import os
import numpy as np

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state


class MNIST:
    def __init__(self, train_perc=0.8, cache_loc="/tmp/mnist_cache"):
        self.train_perc = train_perc
        self.cache_loc = cache_loc
        if not self.checkcache():
            self.download()
        self.load()
    
    def checkcache(self):
        return os.path.exists(self.cache_loc) and os.path.isdir(self.cache_loc)

    def load(self):
        self.x_train = np.load(self.cache_loc+"/x_train.npy")
        self.y_train = np.load(self.cache_loc+"/y_train.npy", allow_pickle=True)
        self.x_test = np.load(self.cache_loc+"/x_test.npy")
        self.y_test = np.load(self.cache_loc+"/y_test.npy", allow_pickle=True)

    
    def download(self):
        print("Downloading dataset...")
        x, y = fetch_openml('mnist_784', version=1, return_X_y=True)
        print("Done.")
        random_state = check_random_state(0)
        permutation = random_state.permutation(x.shape[0])
        x = x[permutation]
        y = y[permutation]
        x = x.reshape((x.shape[0], -1))
        train_samples = int(x.shape[0]*self.train_perc)
        test_samples = x.shape[0] - train_samples
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, train_size=train_samples, test_size=test_samples)
        
        os.mkdir(self.cache_loc)
        np.save(self.cache_loc+"/x_train.npy", x_train)
        np.save(self.cache_loc+"/y_train.npy", y_train)
        np.save(self.cache_loc+"/x_test.npy", x_test)
        np.save(self.cache_loc+"/y_test.npy", y_test)


    def scale(self, max):
        self.x_train = (self.x_train/255)*max
        self.x_test = (self.x_test/255)*max

    def get_train(self):
        return self.x_train, self.y_train
    
    def get_test(self):
        return self.x_test, self.y_test


MNIST()
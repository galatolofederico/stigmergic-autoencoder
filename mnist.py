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
        if not self.load():
            self.download()
            self.load()
    
    def load(self):
        if os.path.exists(self.cache_loc) and os.path.isdir(self.cache_loc):
            self.x_train = np.load(self.cache_loc+"/x_train")
            self.y_train = np.load(self.cache_loc+"/y_train")
            self.x_test = np.load(self.cache_loc+"/x_test")
            self.y_test = np.load(self.cache_loc+"/y_test")
            return True
        return False
    
    def download(self):
        print("Downloading dataset...")
        x, y = fetch_openml('mnist_784', version=1, return_X_y=True)
        print("Done.")
        random_state = check_random_state(0)
        permutation = random_state.permutation(X.shape[0])
        x = x[permutation]
        y = y[permutation]
        x = x.reshape((X.shape[0], -1))
        train_samples = int(X.shape[0]*self.train_perc)
        test_samples = X.shape[0] - train_samples
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, train_size=train_samples, test_size=test_samples)
        
        np.save(self.cache_loc+"/x_train", x_train)
        np.save(self.cache_loc+"/y_train", y_train)
        np.save(self.cache_loc+"/x_test", x_test)
        np.save(self.cache_loc+"/y_test", y_test)



MNIST()


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

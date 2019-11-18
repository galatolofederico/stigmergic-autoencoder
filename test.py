from stigenv import StigAutoEncoderEnv
from mnist import MNIST

dataset = MNIST()
ae = StigAutoEncoderEnv(size=10, dataset=dataset)


print(dataset.get_train()[0][0])
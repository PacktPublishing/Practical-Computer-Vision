from __future__ import print_function

from keras.datasets import cifar10
import matplotlib.pyplot as plt 

# Download and load dataset 
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
# to know the size of data
print("Train data shape:", x_train.shape, "Test data shape:", x_test.shape)

# plot sample image
idx = 1500
print("Label:",labels[y_train[idx][0]])
plt.imshow(x_train[idx])
plt.axis('off')
plt.show()
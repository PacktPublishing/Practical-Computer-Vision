from __future__ import print_function

from keras.datasets import mnist
import matplotlib.pyplot as plt 

# Download and load dataset 
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# to know the size of data
print("Train data shape:", x_train.shape, "Test data shape:", x_test.shape)

# plot sample image
idx = 0
print("Label:",y_train[idx])
plt.imshow(x_train[idx], cmap='gray')
plt.axis('off')
plt.show()
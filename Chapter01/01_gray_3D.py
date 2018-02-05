import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2


# loads and read an image from path to file
img =  cv2.imread('../figures/building_sm.png')

# convert the color to grayscale 
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.resize(gray, (160, 120))

# apply smoothing operation
gray = cv2.blur(gray,(3,3))

# create grid to plot using numpy
xx, yy = np.mgrid[0:gray.shape[0], 0:gray.shape[1]]

# create the figure
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(xx, yy, gray ,rstride=1, cstride=1, cmap=plt.cm.gray,
        linewidth=1)
# show it
plt.show()

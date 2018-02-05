import numpy as np 
import matplotlib.pyplot as plt 
import cv2 
# With jupyter notebook uncomment below line 
# %matplotlib inline 
# This plots figures inside the notebook



def point_operation(img, K, L):
    """
    Applies point operation to given grayscale image
    """
    img = np.asarray(img, dtype=np.float)
    img = img*K  + L
    img[img > 255] = 255
    img[img < 0] = 0
    return np.asarray(img, dtype = np.int)

def main():
    # read an image 
    img = cv2.imread('../figures/flower.png')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # k = 0.5, l = 0
    out1 = point_operation(gray, 0.5, 0)

    # k = 1., l = 10
    out2 = point_operation(gray, 1., 10)

    # k = 0.8, l = 15
    out3 = point_operation(gray, 0.7, 25)
    
    res = np.hstack([gray,out1, out2, out3])
    plt.imshow(res, cmap='gray')
    plt.axis('off')

    plt.show()


if __name__ == '__main__':
    main()
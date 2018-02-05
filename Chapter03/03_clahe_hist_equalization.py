import numpy as np 
import matplotlib.pyplot as plt 
import cv2 
# With jupyter notebook uncomment below line 
# %matplotlib inline 
# This plots figures inside the notebook

def plot_gray(input_image):     
    """     
    plot grayscale image with no axis     
    """  
    # plot grayscale image with gray colormap   
    plt.imshow(input_image, cmap='gray')     
    
    # turn off axis for easier view
    plt.axis('off')
    plt.show()

def plot_hist_cdf(cdf_normalized, img):
    plt.plot(cdf_normalized, color = 'b')
    plt.hist(img.flatten(),256,[0,256], color = 'r')
    plt.xlim([0,256])
    plt.legend(('cdf','histogram'), loc = 'upper left')
    plt.show()
    

    
def main():
    # read an image 
    img = cv2.imread('../figures/flower.png')
    crop_gray = cv2.cvtColor(img[100:400, 100:400], cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl1 = clahe.apply(crop_gray)
    res = np.hstack((crop_gray,cl1))
    plot_gray(res)
    
if __name__ == '__main__':
    main()
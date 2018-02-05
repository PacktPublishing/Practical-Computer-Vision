import numpy as np 
import matplotlib.pyplot as plt 
import cv2 
# With jupyter notebook uncomment below line 
# %matplotlib inline 
# This plots figures inside the notebook

def plot_cv_img(input_image, output_image):     
    """     
    Converts an image from BGR to RGB and plots     
    """   
    # change color channels order for matplotlib     
    fig, ax = plt.subplots(nrows=1, ncols=2)

    ax[0].imshow(input_image, cmap='gray')          
    ax[0].set_title('Input Image')
    ax[0].axis('off')
    
    ax[1].imshow(output_image, cmap='gray')          
    ax[1].set_title('Convolution ')
    ax[1].axis('off')    
    
    plt.savefig('../figures/03_convolution.png')

    plt.show()


def main():
    # read an image 
    img = cv2.imread('../figures/flower.png')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # initialize noise image with zeros
    noise = np.zeros((400, 600))

    # fill the image with random numbers in given range
    cv2.randu(noise, 0, 255)
    
    noisy_gray = gray + np.array(0.2*noise, dtype=np.int)
    
    kernel = np.ones((5,5),np.float32)/25
    dst = cv2.filter2D(gray,-1,kernel)
    
    # Do plot
    plot_cv_img(gray, dst)

if __name__ == '__main__':
    main()
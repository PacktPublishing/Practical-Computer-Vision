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

    
def plot_dft(crop_gray, magnitude_spectrum):
    plt.subplot(121),plt.imshow(crop_gray, cmap = 'gray')
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
    plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
    plt.show()
    

    
def main():
    # read an image 
    img = cv2.imread('../figures/flower.png')
    
    # create cropped grayscale image from the original image
    crop_gray = cv2.cvtColor(img[100:400, 100:400], cv2.COLOR_BGR2GRAY)
    
    # take discrete fourier transform 
    dft = cv2.dft(np.float32(crop_gray),flags = cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))
    
    # plot results
    plot_dft(crop_gray, magnitude_spectrum)
    
   
    
if __name__ == '__main__':
    main()
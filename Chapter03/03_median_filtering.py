import numpy as np 
import matplotlib.pyplot as plt 
import cv2 
# With jupyter notebook uncomment below line 
# %matplotlib inline 
# This plots figures inside the notebook

def plot_cv_img(input_image, output_image1, output_image2, output_image3):     
    """     
    Converts an image from BGR to RGB and plots     
    """   

    fig, ax = plt.subplots(nrows=1, ncols=4)

    ax[0].imshow(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB))          
    ax[0].set_title('Input Image')
    ax[0].axis('off')
    
    ax[1].imshow(cv2.cvtColor(output_image1, cv2.COLOR_BGR2RGB))          
    ax[1].set_title('Median Filter (3,3)')
    ax[1].axis('off')

    ax[2].imshow(cv2.cvtColor(output_image2, cv2.COLOR_BGR2RGB))          
    ax[2].set_title('Median Filter (5,5)')
    ax[2].axis('off')    

    ax[3].imshow(cv2.cvtColor(output_image3, cv2.COLOR_BGR2RGB))          
    ax[3].set_title('Median Filter (7,7)')
    ax[3].axis('off')
    
    # plt.savefig('../figures/03_median_filter.png')

    plt.show()
    

def main():
    # read an image 
    img = cv2.imread('../figures/flower.png')

    
    median1 = cv2.medianBlur(img,3)
    median2 = cv2.medianBlur(img,5)
    median3 = cv2.medianBlur(img,7)
    
    
    # Do plot
    plot_cv_img(img, median1, median2, median3)

if __name__ == '__main__':
    main()
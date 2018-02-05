import numpy as np 
import matplotlib.pyplot as plt 
import cv2 
# With jupyter notebook uncomment below line 
# %matplotlib inline 
# This plots figures inside the notebook

def plot_lr_img(input_image, l1, l2, l3):     
    """     
    Converts an image from BGR to RGB and plots     
    """   

    fig, ax = plt.subplots(nrows=1, ncols=4)

    ax[0].imshow(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB))          
    ax[0].set_title('Input Image (400,600) ')
    ax[0].axis('off')
    
    ax[1].imshow(cv2.cvtColor(l1, cv2.COLOR_BGR2RGB))          
    ax[1].set_title('Lower Resolution (200, 300)')
    ax[1].axis('off')

    ax[2].imshow(cv2.cvtColor(l2, cv2.COLOR_BGR2RGB))          
    ax[2].set_title('Lower Resolution (100, 150)')
    ax[2].axis('off')    

    ax[3].imshow(cv2.cvtColor(l3, cv2.COLOR_BGR2RGB))          
    ax[3].set_title('Lower Resolution (50, 75)')
    ax[3].axis('off')
    
    # plt.savefig('../figures/03_pyr_down_sample.png')

    plt.show()
    

def plot_hy_img(input_image, h1, h2, h3):     
    """     
    Converts an image from BGR to RGB and plots     
    """   

    fig, ax = plt.subplots(nrows=1, ncols=4)

    ax[0].imshow(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB))          
    ax[0].set_title('Input Image (50,75) ')
    ax[0].axis('off')
    
    ax[1].imshow(cv2.cvtColor(h1, cv2.COLOR_BGR2RGB))          
    ax[1].set_title('Higher Resolution (100, 150)')
    ax[1].axis('off')

    ax[2].imshow(cv2.cvtColor(h2, cv2.COLOR_BGR2RGB))          
    ax[2].set_title('Higher Resolution (200, 300)')
    ax[2].axis('off')    

    ax[3].imshow(cv2.cvtColor(h3, cv2.COLOR_BGR2RGB))          
    ax[3].set_title('Higher Resolution (400, 600)')
    ax[3].axis('off')
    
    # plt.savefig('../figures/03_pyr_down_sample.png')

    plt.show()


def main():
    # read an image 
    img = cv2.imread('../figures/flower.png')
    print(img.shape)

    lower_resolution1 = cv2.pyrDown(img)
    print(lower_resolution1.shape)

    lower_resolution2 = cv2.pyrDown(lower_resolution1)
    print(lower_resolution2.shape)

    lower_resolution3 = cv2.pyrDown(lower_resolution2)
    print(lower_resolution3.shape)

    higher_resolution3 = cv2.pyrUp(lower_resolution3)
    print(higher_resolution3.shape)

    higher_resolution2 = cv2.pyrUp(higher_resolution3)
    print(higher_resolution2.shape)

    higher_resolution1 = cv2.pyrUp(higher_resolution2)
    print(higher_resolution1.shape)




    
    # Do plot
    plot_lr_img(img, lower_resolution1, lower_resolution2, lower_resolution3)
    plot_hy_img(lower_resolution3, higher_resolution3, higher_resolution2, higher_resolution1)

if __name__ == '__main__':
    main()
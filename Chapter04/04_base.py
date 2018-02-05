import numpy as np 
import matplotlib.pyplot as plt 
import cv2 
# With jupyter notebook uncomment below line 
# %matplotlib inline 
# This plots figures inside the notebook

def plot_cv_img(input_image1, input_image2, input_image3):     
    """     
    Converts an image from BGR to RGB and plots     
    """   
    # change color channels order for matplotlib     
    fig, ax = plt.subplots(nrows=1, ncols=3)
    input_image1 = cv2.cvtColor(input_image1,cv2.COLOR_BGR2RGB)
    input_image2 = cv2.cvtColor(input_image2,cv2.COLOR_BGR2RGB)
    input_image3 = cv2.cvtColor(input_image3,cv2.COLOR_BGR2RGB)


    ax[0].imshow(input_image1)          
    ax[0].set_title('Image1')
    ax[0].axis('off')
    
    ax[1].imshow(input_image2)          
    ax[1].set_title('Image2')
    ax[1].axis('off') 

    ax[2].imshow(input_image3)          
    ax[2].set_title('Image3')
    ax[2].axis('off')
    
    plt.savefig('../figures/04_harris_corners1.png')

    plt.show()

def compute_harris_corners(input):
    gray = cv2.cvtColor(input,cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray,2,5,0.04)
    #result is dilated for marking the corners, not important
    dst = cv2.dilate(dst,None)
    # Threshold for an optimal value, it may vary depending on the image.
    input[dst>0.01*dst.max()]=[0,255,0]
    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(input, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

def display_harris_corners(input_img):
    """
    computes corners in colored image and plot it.
    """
    # first convert to grayscale with float32 values
    gray = cv2.cvtColor(input_img,cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    
    # using opencv harris corner implementation
    corners = cv2.cornerHarris(gray,2,7,0.04)
    
#     # result is dilated for marking the corners, not important
#     dst = cv2.dilate(dst,None)
    
    # additional thresholding and marking corners for plotting
    input_img[corners>0.01*corners.max()]=[255,0,0]
    
    return input_img
    # # plot image
    # plt.figure(figsize=(12, 8))
    # plt.imshow(cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB))
    # plt.axis('off')

def main():
    # read an image 
    img1 = cv2.imread('../figures/building2.jpg')
    img2 = cv2.imread('../figures/flower.png')
    img3 = cv2.imread('../figures/building_crop.jpg')
    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    

    # compute harris corners and display 
    out1 = display_harris_corners(img1)
    out2 = display_harris_corners(img2)
    out3 = display_harris_corners(img3)

    
    # Do plot
    plot_cv_img(out1, out2, out3)

if __name__ == '__main__':
    main()
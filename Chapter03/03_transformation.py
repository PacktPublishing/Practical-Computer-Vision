import numpy as np 
import matplotlib.pyplot as plt 
import cv2 
# With jupyter notebook uncomment below line 
# %matplotlib inline 
# This plots figures inside the notebook

def plot_cv_img(input_image):     
    """     
    Converts an image from BGR to RGB and plots     
    """   
    # change color channels order for matplotlib     
    plt.imshow(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB))          

    # For easier view, turn off axis around image     
    plt.axis('off')
    plt.show()
    

def main():
    # read an image 
    img = cv2.imread('../figures/flower.png')
    
    # create transformation matrix 
    translation_matrix = np.float32([[1,0,160],[0,1,40]])
    transformed = cv2.warpAffine(img, translation_matrix, (img.shape[1],img.shape[0]))

    # Do plot
    plot_cv_img(transformed)

if __name__ == '__main__':
    main()
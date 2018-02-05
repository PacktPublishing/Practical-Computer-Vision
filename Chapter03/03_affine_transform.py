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
    img = cv2.imread('../figures/building.jpg')
    
    # create transformation matrix form preselected points
    pts1 = np.float32([[50,50],[200,50],[50,200]])
    pts2 = np.float32([[10,100],[200,50],[100,250]])
    affine_tr = cv2.getAffineTransform(pts1,pts2)

    transformed = cv2.warpAffine(img, affine_tr, (img.shape[1]*2,img.shape[0]*2))

    # Do plot
    plot_cv_img(transformed)

if __name__ == '__main__':
    main()
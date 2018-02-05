import numpy as np 
import matplotlib.pyplot as plt 
import cv2 
# With jupyter notebook uncomment below line 
# %matplotlib inline 
# This plots figures inside the notebook



def plot_imgs(img1, img2):     
    """     
    Converts an image from BGR to RGB and plots     
    """   

    fig, ax = plt.subplots(nrows=1, ncols=2)

    ax[0].imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))          
    ax[0].set_title('FAST points on Image (th=10)')
    ax[0].axis('off')
    
    ax[1].imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))          
    ax[1].set_title('FAST points on Image (th=30)')
    ax[1].axis('off')

    # ax[2].imshow(cv2.cvtColor(img3, cv2.COLOR_BGR2RGB))          
    # ax[2].set_title('FAST Points(th=15)')
    # ax[2].axis('off')    

    # ax[3].imshow(cv2.cvtColor(img4, cv2.COLOR_BGR2RGB))          
    # ax[3].set_title('FAST Points(th=50)')
    # ax[3].axis('off')
    
    # plt.savefig('../figures/04_fast_features_thres.png')

    plt.show()

def compute_fast_det(filename, is_nms=True, thresh = 10):

    img = cv2.imread(filename)
    
    # Initiate FAST object with default values
    fast = cv2.FastFeatureDetector_create() #FastFeatureDetector()

    # find and draw the keypoints
    if not is_nms:
        fast.setNonmaxSuppression(0)

    fast.setThreshold(thresh)

    kp = fast.detect(img,None)
    cv2.drawKeypoints(img, kp, img, color=(255,0,0))
    
    return img


def main():
    # read an image 
    #img = cv2.imread('../figures/flower.png')
    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    filename1 = '../figures/flower.png'
    filename2 = '../figures/building_sm.png'
    filename3 = '../figures/outdoor.jpg'
    # compute harris corners and display 
    img1 = compute_fast_det(filename1, thresh = 10)
    img2 = compute_fast_det(filename2, thresh = 30)
    #img3 = compute_fast_det(filename, thresh = 10)
    #img4 = compute_fast_det(filename, thresh = 10)
    
    # Do plot
    plot_imgs(img1, img2)

if __name__ == '__main__':
    main()
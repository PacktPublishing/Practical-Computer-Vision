import numpy as np 
import matplotlib.pyplot as plt 
import cv2 
print(cv2.__version__)
# With jupyter notebook uncomment below line 
# %matplotlib inline 
# This plots figures inside the notebook




def compute_orb_keypoints(filename):
    """
    Takes in filename to read and computes ORB keypoints
    Returns image, keypoints and descriptors 
    """

    img = cv2.imread(filename)
    img = cv2.pyrDown(img)
    img = cv2.pyrDown(img)
    # img = cv2.pyrDown(img)
    # img = cv2.pyrDown(img)
    # create orb object
    orb = cv2.ORB_create()
    
    # set parameters 
    orb.setScoreType(cv2.FAST_FEATURE_DETECTOR_TYPE_9_16)
    orb.setWTA_K(3)
    
    kp = orb.detect(img,None)

    kp, des = orb.compute(img, kp)
    return img,kp,  des


def draw_keyp(img, kp):
    """
    Draws color around keypoint pixels
    """
    cv2.drawKeypoints(img,kp,img, color=(255,0,0), flags=2) 
    return img


def plot_orb(filename):
    """
    Plots ORB keypoints from filename
    """
    img,kp, des = compute_orb_keypoints(filename)
    img = draw_keyp(img, kp)
    plot_img(img)


def plot_img(img):
    """
    Generic plotting of opencv image
    """
    fig = plt.figure(figsize=(16,12))
    ax = fig.add_subplot(1,1,1)
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

def compute_img_matches(filename1, filename2, thres=10):
    """
    Extracts ORB features from given filenames
    Computes ORB matches and plot them side by side 
    """
    img1, kp1, des1 = compute_orb_keypoints(filename1)
    img2, kp2, des2 = compute_orb_keypoints(filename2)
    
    matches = brute_force_matcher(des1, des2)
    draw_matches(img1, img2, kp1, kp2, matches, thres)
    
def brute_force_matcher(des1, des2):
    """
    Brute force matcher to match ORB feature descriptors
    """
    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING2, crossCheck=True)
    # Match descriptors.
    matches = bf.match(des1,des2)

    # Sort them in the order of their distance.
    matches = sorted(matches, key = lambda x:x.distance)

    return matches

def draw_matches(img1, img2, kp1, kp2, matches, thres=10):
    """
    Utility function to draw lines connecting matches between two images.
    """
    draw_params = dict(matchColor = (0,255,0),
                       singlePointColor = (255,0,0),
                       flags = 0)

    # Draw first thres matches.
    img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:thres],None, **draw_params)
    plot_img(img3)



def main():
    # read an image 
    filename2 = '../figures/building_7.JPG'
    filename1 = '../figures/building_crop.jpg'
    compute_img_matches(filename1, filename2, thres=20)
   


if __name__ == '__main__':
    main()


import numpy as np 
import matplotlib.pyplot as plt 
import cv2 
print(cv2.__version__)
import glob
# With jupyter notebook uncomment below line 
# %matplotlib inline 
# This plots figures inside the notebook



def compute_orb_keypoints(filename):
    """
    Reads image from filename and computes ORB keypoints
    Returns image, keypoints and descriptors. 
    """
    # load image
    img = cv2.imread(filename)
    
    # create orb object
    orb = cv2.ORB_create()
    
    # set parameters 
    orb.setScoreType(cv2.FAST_FEATURE_DETECTOR_TYPE_9_16)
    orb.setWTA_K(3)
    
    # detect keypoints
    kp = orb.detect(img,None)

    # for detected keypoints compute descriptors. 
    kp, des = orb.compute(img, kp)
    
    return img,kp, des

    
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

def compute_fundamental_matrix(filename1, filename2):
    """
    Takes in filenames of two input images 
    Return Fundamental matrix computes 
    using 8 point algorithm
    """
    # compute ORB keypoints and descriptor for each image
    img1, kp1, des1 = compute_orb_keypoints(filename1)
    img2, kp2, des2 = compute_orb_keypoints(filename2)
    
    # compute keypoint matches using descriptor
    matches = brute_force_matcher(des1, des2)
    
    # extract points
    pts1 = []
    pts2 = []
    for i,(m) in enumerate(matches):
        if m.distance < 20:
            #print(m.distance)
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)
    pts1  = np.asarray(pts1)
    pts2 = np.asarray(pts2)
    
    # Compute fundamental matrix
    F, mask = cv2.findFundamentalMat(pts1,pts2,cv2.FM_8POINT)
    return F


def main():
    # read list of images form dir in sorted order 
    image_dir = '/Users/mac/Documents/dinoRing/'
    file_list = sorted(glob.glob(image_dir+'*.png'))

    #compute F matrix between two images
    print(compute_fundamental_matrix(file_list[0], file_list[2]))
   


if __name__ == '__main__':
    main()


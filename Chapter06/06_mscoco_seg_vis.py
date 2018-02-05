import numpy as np
import os
import sys
# import tensorflow as tf
import cv2
from matplotlib import pyplot as plt
# inside jupyter uncomment next line 
# %matplotlib inline
import random
import time
from pycocotools.coco import COCO
import skimage.io as io


def draw_segmentation_mask(img,anns):
    for ann in anns:
        for seg in ann['segmentation']:
            poly = np.array(seg).reshape((int(len(seg)/2), 2))
        cv2.fillConvexPoly(img,np.int32([poly]), color=(255, 255, 255) )
    return cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)

def draw_segmentation_boundary(img,anns):
    for ann in anns:
        for seg in ann['segmentation']:
            poly = np.array(seg).reshape((int(len(seg)/2), 2))
        cv2.polylines(img, np.int32([poly]), True, color=(0, 255, 0))
    return img

def main():
    annFile='annotations/instances_train2017.json'
    coco=COCO(annFile)
    # display COCO categories and supercategories
    cats = coco.loadCats(coco.getCatIds())
    nms=[cat['name'] for cat in cats]
    print('COCO categories: \n{}\n'.format(' '.join(nms)))

    nms = set([cat['supercategory'] for cat in cats])
    print('COCO supercategories: \n{}'.format(' '.join(nms)))

    # get all images containing given categories, select one at random
    catIds = coco.getCatIds(catNms=['person','dog']);
    imgIds = coco.getImgIds(catIds=catIds );
    # imgIds = coco.getImgIds(imgIds = [324158])
    img_meta = coco.loadImgs(imgIds[np.random.randint(0,len(imgIds))])[0]
    
    I = io.imread(img_meta['coco_url'])
    cv_img = I.copy()
    plt.imshow(cv_img)
    plt.axis('off')
    plt.show()

    annIds = coco.getAnnIds(imgIds=img_meta['id'], catIds=catIds, iscrowd=None)
    anns = coco.loadAnns(annIds)
    # print(anns)

    # create mask of zero
    mask = np.zeros(cv_img.shape, dtype=np.uint8)
    mask = draw_segmentation_boundary(mask, anns)
    plt.imshow(mask, cmap='gray')
    plt.axis('off')
    plt.show()
    

if __name__=='__main__':
    main()


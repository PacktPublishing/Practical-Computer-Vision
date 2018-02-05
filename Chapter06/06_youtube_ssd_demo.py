import numpy as np
import cv2
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import pafy
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
import random
import time
sys.path.append("..")
from utils import label_map_util
from utils import visualization_utils as vis_util

# What model to download.
# ssd_inception_v2_coco_2017_11_17
# faster_rcnn_inception_v2_coco_2017_11_08
MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
# MODEL_NAME = 'faster_rcnn_inception_v2_coco_2017_11_08'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90

# create youtube capture
url = 'https://www.youtube.com/watch?v=fq-X9UZMLRk'
videoPafy = pafy.new(url)



def load_label_dict(PATH_TO_LABELS):
  label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
  categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
  category_index = label_map_util.create_category_index(categories)
  #print(category_index)
  return category_index 






def show_cv_img_with_detections(img, dets,scores, classes,thres=0.4):
    height = img.shape[0]
    width = img.shape[1]
    colors = dict()
    
    for i in range(dets.shape[0]):

        cls_id = int(classes[i])
        # print(cls_id)
        if cls_id >= 0:
            score = scores[i]

            if score > thres:
                if cls_id not in colors:
                  colors[cls_id] = (random.random(), random.random(), random.random())
                xmin = int(dets[i, 1] * width)
                ymin = int(dets[i, 0] * height)
                xmax = int(dets[i, 3] * width)
                ymax = int(dets[i, 2] * height)

                # print(xmin, ymin, xmax, ymax)
                cv2.rectangle(img, (xmin, ymin), (xmax, ymax), colors[cls_id])
                class_name = str(category_index[cls_id]['name'])
                cv2.putText(img, '{:s} {:.3f}'.format(class_name, score), (xmin, ymin), cv2.FONT_HERSHEY_PLAIN, 0.5, colors[cls_id])
                
    return img


def show_mpl_img_with_detections(img, dets,scores, classes,thres=0.6):
    
    import matplotlib.pyplot as plt
    import random
    plt.figure(figsize=(8,6))
    plt.imshow(img)
    height = img.shape[0]
    width = img.shape[1]
    colors = dict()
    # dets = dets[0]
    # print(dets.shape)
    for i in range(dets.shape[0]):

        cls_id = int(classes[i])
        # print(cls_id)
        if cls_id >= 0:
            score = scores[i]

            if score > thres:
                if cls_id not in colors:
                  colors[cls_id] = (random.random(), random.random(), random.random())
                xmin = int(dets[i, 1] * width)
                ymin = int(dets[i, 0] * height)
                xmax = int(dets[i, 3] * width)
                ymax = int(dets[i, 2] * height)
                rect = plt.Rectangle((xmin, ymin), xmax - xmin,
                                     ymax - ymin, fill=False,
                                     edgecolor=colors[cls_id],
                                     linewidth=2.5)
                plt.gca().add_patch(rect)
                class_name = str(category_index[cls_id]['name'])
                
                plt.gca().text(xmin, ymin - 2,
                               '{:s} {:.3f}'.format(class_name, score),
                               bbox=dict(facecolor=colors[cls_id], alpha=0.5),
                               fontsize=8, color='white')
    plt.axis('off')
    plt.savefig(filename)
    plt.close()
    plt.pause(0.001)
    # cv2.imwrite(filename,img)
    return 

# download model 
opener = urllib.request.URLopener()
opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
tar_file = tarfile.open(MODEL_FILE)
for file in tar_file.getmembers():
  file_name = os.path.basename(file.name)
  if 'frozen_inference_graph.pb' in file_name:
    tar_file.extract(file, os.getcwd())

# import frozen graph
detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

# load labels
category_index = load_label_dict(PATH_TO_LABELS)


best = videoPafy.getbest(preftype="webm")
cap = cv2.VideoCapture(videoPafy.videostreams[2].url)
# cap = cv2.VideoCapture(best.url)
# run session
with detection_graph.as_default():
  with tf.Session(graph=detection_graph) as sess:
    # g = tf.get_default_graph()
    # print(g.get_operations())

    # Definite input and output Tensors for detection_graph
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    # Each box represents a part of the image where a particular object was detected.
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')
    skip = 5
    while(True):
      # Capture frame-by-frame
      ret, frame = cap.read()
      if skip != 0:
        skip -=1
      skip = 5 
      frame_bgr = frame
      frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)



      # the array based representation of the image will be used later in order to prepare the
      # result image with boxes and labels on it.
      # image_np = load_image_into_numpy_array(frame)
      image_np = np.asarray(frame,dtype=np.uint8)
      # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
      image_np_expanded = np.expand_dims(image_np, axis=0)
      # Actual detection.
      # (boxes, scores, classes, num) = sess.run(
      #     [detection_boxes, detection_scores, detection_classes, num_detections],
      #     feed_dict={image_tensor: image_np_expanded})
      # print(classes)
      # out = show_cv_img_with_detections(frame_bgr, boxes[0],scores[0], classes[0], thres=0.45)
      
      cv2.imshow('frame',frame_bgr)
      if cv2.waitKey(1) & 0xFF == ord('q'):
        
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
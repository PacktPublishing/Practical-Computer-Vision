from keras.applications.resnet50 import ResNet50
import numpy as np 
import cv2 
from keras.applications.resnet50 import preprocess_input, decode_predictions
import time

def get_model():
    """
    Loads Resnet and prints model structure
    """
    
    # create model 
    model = ResNet50(weights='imagenet')

    # To print our model loaded
    model.summary()
    return model

def preprocess_img(img):
    # apply opencv preprocessing
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img,  (224, 224)) 
    img = img[np.newaxis, :, :, :]
    img = np.asarray(img, dtype=np.float)
    
    # further use imagenet specific preprocessing
    # this applies color channel specific mean normalization
    x = preprocess_input(img)
    print(x.shape)
    return x

# read input image and preprocess
img = cv2.imread('../figures/train1.png')
input_x = preprocess_img(img)

# create model with pre-trained weights
resnet_model = get_model()

# run predictions only , no training
start = time.time()
preds = resnet_model.predict(input_x)
print(time.time() - start)

# decode prediction to index of classes, top 5 predictions
print('Predicted:', decode_predictions(preds, top=5)[0])
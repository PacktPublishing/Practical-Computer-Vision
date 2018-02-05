from keras.models import *
from keras.layers import *
from keras.applications.vgg16 import VGG16

def create_model_fcn32(nb_class, input_w=256):
    """
    Create FCN-32s model for segmentaiton. 
    Input:
        nb_class: number of detection categories
        input_w: input width, using square image

    Returns model created for training. 
    """
    input = Input(shape=(input_w, input_w, 3))

    # initialize feature extractor excuding fully connected layers
    # here we use VGG model, with pre-trained weights. 
    vgg = VGG16(include_top=False, weights='imagenet', input_tensor=input)
    # create further network
    x = Conv2D(4096, kernel_size=(7,7), use_bias=False,
               activation='relu', padding="same")(vgg.output)
    x = Dropout(0.5)(x)
    x = Conv2D(4096, kernel_size=(1,1), use_bias=False,
               activation='relu', padding="same")(x)
    x = Dropout(0.5)(x)
    x = Conv2D(nb_class, kernel_size=(1,1), use_bias=False, 
               padding="same")(x)
    # upsampling to image size
    x = Conv2DTranspose(nb_class , 
                        kernel_size=(64,64), 
                        strides=(32,32), 
                        use_bias=False, padding='same')(x)
    
    
    x = Activation('softmax')(x)
    model = Model(input, x)
    model.summary()
    return model

# Create model for pascal voc image segmentation for 21 classes
model = create_model_fcn32(21)

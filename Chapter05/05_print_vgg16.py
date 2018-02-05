from keras.applications.vgg16 import VGG16

def print_model():
    """
    Loads VGGNet and prints model structure
    """
    
    # create model 
    model = VGG16(weights='imagenet')

    # prints our model created
    model.summary()

print_model()
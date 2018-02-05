from keras.applications.inception_v3 import InceptionV3

def print_model():
    """
    Loads Inceptionv3 and prints model structure
    """
    
    # create model 
    model = InceptionV3(weights='imagenet')

    # prints our model created
    model.summary()

print_model()
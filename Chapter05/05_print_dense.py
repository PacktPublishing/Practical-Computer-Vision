from keras.layers import Dense, Input
from keras.models import Model

def print_model():
    """
    Creates a sample model and prints output shape
    Use this to analyse dense/Fully Connected parameters
    """
    # create input with given shape 
    x = Input(shape=(512,))

    # create a fully connected layer layer
    y = Dense(32)(x)
    
    # create model 
    model = Model(inputs=x, outputs=y)

    # prints our model created
    model.summary()

print_model()
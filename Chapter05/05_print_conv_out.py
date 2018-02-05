from keras.layers import Conv2D, Input
from keras.models import Model

def print_model():
    """
    Creates a sample model and prints output shape
    Use this to analyse convolution parameters
    """
    # create input with given shape 
    x = Input(shape=(512,512,3))

    # create a convolution layer
    y = Conv2D(filters=32, 
               kernel_size=(5,5), 
               strides=1, padding="same",
               use_bias=True)(x)
    
    # create model 
    model = Model(inputs=x, outputs=y)

    # prints our model created
    model.summary()

print_model()
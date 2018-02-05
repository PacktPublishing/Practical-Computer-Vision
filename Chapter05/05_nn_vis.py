import keras 
import keras.backend as K
from keras.layers import Dense, Conv2D, Input, MaxPooling2D, Flatten
from keras.models import Model
from keras.datasets import fashion_mnist
from keras.callbacks import ModelCheckpoint


# setup parameters
batch_sz = 128 
nb_class = 10 
nb_epochs = 10 

img_h, img_w = 28, 28 
print( K.image_data_format())

# input image dimensions
img_rows, img_cols = 28, 28

def get_dataset():
    """
    Return processed and reshaped dataset for training
    In this cases Fashion-mnist dataset.
    """
    # load mnist dataset
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    
    # test and train datasets
    print("Nb Train:", x_train.shape[0], "Nb test:",x_test.shape[0])
    x_train = x_train.reshape(x_train.shape[0], img_h, img_w, 1)
    x_test = x_test.reshape(x_test.shape[0], img_h, img_w, 1)
    in_shape = (img_h, img_w, 1)

    # normalize inputs
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255.0
    x_test /= 255.0

    # convert to one hot vectors 
    y_train = keras.utils.to_categorical(y_train, nb_class)
    y_test = keras.utils.to_categorical(y_test, nb_class)
    return x_train, x_test, y_train, y_test

x_train, x_test, y_train, y_test = get_dataset()

def create_model(img_h=28, img_w=28):
    inputs = Input(shape=(img_h, img_w, 1))
    x = Conv2D(32, kernel_size=(3,3), activation='relu')(inputs) # 32C 3K 1S VP RELU
    x = Conv2D(32, kernel_size=(3,3), activation='relu')(x) # 64C 3K 1S VP RELU
    x = MaxPooling2D(pool_size=(2,2))(x) # pool2
    x = Conv2D(64, kernel_size=(3,3), activation='relu')(x) # 32C 3K 1S VP RELU
    x = Conv2D(64, kernel_size=(3,3), activation='relu')(x) # 64C 3K 1S VP RELU
    x = MaxPooling2D(pool_size=(2,2))(x) # pool2
    x = Flatten()(x)
    preds = Dense(nb_class, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=preds)
    print(model.summary())
    return model

model = create_model()

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.SGD(lr=0.001),
              metrics=['accuracy'])

callback = ModelCheckpoint()

# start training
model.fit(x_train, y_train,
          batch_size=batch_sz,
          epochs=nb_epochs,
          verbose=1,
          validation_data=(x_test, y_test), callbacks=[callback])

# Evaluate
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# import libraries
from keras.layers import Input, Conv2D, MaxPooling2D, Activation, add, BatchNormalization
from keras.models import Model

inputs = Input(shape = (32, 32, 3))

# first convolutional layer
x = Conv2D(64, (3, 3), padding = 'same')(inputs)
x = Activation('relu')(x)

# create layers
for i in range(3):
    x = Conv2D(64, (3, 3), padding = 'same')(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
 
    x = MaxPooling2D((2, 2))(x)
  
    x = Conv2D(10, (3, 3), padding = 'same')(x)
    x = Activation('softmax')(x)

model = Model(inputs = inputs, outputs = x)

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

model.fit(x_train, y_train, batch_size = 64, epochs = 50, validation_data = (x_test, y_test))

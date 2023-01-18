# import libraries
from keras.layers import Input, Conv2D, MaxPooling2D, Activation, add, BatchNormalization, GlobalAveragePooling2D, Dense
from keras.models import Model

inputs = Input(shape = (224, 224, 3))

x = Conv2D(64, (3, 3), padding = 'same')(inputs)
x = Activation('relu')(x)
x = Conv2D(64, (3, 3), strides = (1,1), padding = 'same', groups = 32)(x)
x = Activation('relu')(x)
x = BatchNormalization()(x)

x = MaxPooling2D((2, 2))(x)

x = Conv2D(10, (3, 3), padding = 'same')(x)
x = Activation('softmax')(x)

x = GlobalAveragePooling2D()(x)

x = Dense(num_classes, activation = 'softmax')(x)

model = Model(inputs = inputs, outputs = x)

model.fit(x_train, y_train, batch_size = 64, epochs = 50, validation_data = (x_test, y_test))

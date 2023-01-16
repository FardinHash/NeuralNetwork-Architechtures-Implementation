# import libraries
from keras.applications import ResNet50V2
from keras.layers import Input

# define input shape
input_shape = (224, 224, 3)

# create the base model
base_model = ResNet50V2(weights = 'imagenet',
                      include_top = False,
                      input_shape = input_shape)

# add a global average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)

# add a fully-connected layer
x = Dense(1024, activation  ='relu')(x)

# add a logistic layer
predictions = Dense(num_classes, activation = 'softmax')(x)

# this is the model will be trained
model = Model(inputs = base_model.input, outputs = predictions)

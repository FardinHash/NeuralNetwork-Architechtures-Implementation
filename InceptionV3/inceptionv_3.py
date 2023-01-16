# import libraries
from keras.applications import InceptionV3
from keras.layers import Input
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K

# define input shape
input_shape = (299, 299, 3)

# create the base model
base_model = InceptionV3(weights = 'imagenet', 
                      include_top = False, 
                      input_shape = input_shape)

#freeze the layers
for layer in base_model.layers:
    layer.trainable = False

#create the model
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation = 'relu')(x)
x = Dense(1024, activation = 'relu')(x)
x = Dense(512, activation = 'relu')(x)

# prediction
predictions = Dense(num_classes, activation = 'softmax')(x)
model = Model(inputs = base_model.input, outputs = predictions)

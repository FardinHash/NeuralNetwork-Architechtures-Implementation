# import libraries
from keras.applications import MobileNet
from keras.layers import Input
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D

# define input shape
input_shape = (224, 224, 3)

# create the base model
base_model = MobileNet(weights = 'imagenet', 
                      include_top = False, 
                      input_shape = input_shape,
                      alpha = 1.0)

#freeze the layers
for layer in base_model.layers:
    layer.trainable = False

#create the new model
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation = 'relu')(x)
x = Dense(1024, activation = 'relu')(x)
x = Dense(512, activation = 'relu')(x)

# predictions
predictions = Dense(num_classes, activation = 'softmax')(x)
model = Model(inputs = base_model.input, outputs = predictions)

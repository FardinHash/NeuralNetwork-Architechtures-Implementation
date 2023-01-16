# import libraries
from efficientnet import EfficientNetB3
from keras.layers import Input
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D

# define input shape
input_shape = (300, 300, 3)

# create the base model
base_model = EfficientNetB3(weights = 'imagenet', 
                      include_top = False, 
                      input_shape = input_shape)

# freeze the layers
for layer in base_model.layers:
    layer.trainable = False

# create the new model
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation = 'relu')(x)
x = Dense(1024, activation = 'relu')(x)
x = Dense(512, activation ='relu')(x)

# predictions
predictions = Dense(num_classes, activation = 'softmax')(x)
model = Model(inputs = base_model.input, outputs = predictions)

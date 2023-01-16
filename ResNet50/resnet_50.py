# import libraries
from keras import layers
from keras import models
from keras.applications import ResNet50
from keras.layers import Input

#define input shape
input_shape = (224, 224, 3)

#create the base model
base_model = ResNet50(weights = 'imagenet', 
                      include_top = False, 
                      input_shape = input_shape)

#freeze the layers
for layer in base_model.layers:
    layer.trainable = False

#create the model
x = base_model.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(1024, activation = 'relu')(x)
x = layers.Dropout(0.5)(x)

# predictions
predictions = layers.Dense(num_classes, activation = 'softmax')(x)
model = models.Model(inputs = base_model.input, outputs = predictions)

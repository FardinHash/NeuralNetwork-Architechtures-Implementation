# import libraries
import tensorflow as tf

# Define the model
model = tf.keras.Sequential()

# First Convolutional Layer
model.add(tf.keras.layers.Conv2D(filters = 96, kernel_size = (11, 11), strides = (4, 4), activation = 'relu', input_shape = (224, 224, 3)))
model.add(tf.keras.layers.MaxPool2D(pool_size = (3, 3), strides = (2, 2)))
# Second Convolutional Layer
model.add(tf.keras.layers.Conv2D(filters = 256, kernel_size = (5, 5), activation = 'relu', padding = 'same'))
model.add(tf.keras.layers.MaxPool2D(pool_size = (3, 3), strides = (2, 2)))
# Third Convolutional Layer
model.add(tf.keras.layers.Conv2D(filters = 384, kernel_size = (3, 3), activation = 'relu', padding = 'same'))
# Fourth Convolutional Layer
model.add(tf.keras.layers.Conv2D(filters = 384, kernel_size = (3, 3), activation = 'relu', padding = 'same'))
# Fifth Convolutional Layer
model.add(tf.keras.layers.Conv2D(filters = 256, kernel_size = (3, 3), activation = 'relu', padding = 'same'))
model.add(tf.keras.layers.MaxPool2D(pool_size = (3, 3), strides = (2, 2)))

# Flatten the output from the convolutional layers
model.add(tf.keras.layers.Flatten())

# Fully Connected Layers
model.add(tf.keras.layers.Dense(units = 4096, activation = 'relu'))
model.add(tf.keras.layers.Dropout(rate = 0.5))
model.add(tf.keras.layers.Dense(units = 4096, activation = 'relu'))
model.add(tf.keras.layers.Dropout(rate = 0.5))

# Output Layer
model.add(tf.keras.layers.Dense(units = 1000, activation = 'softmax'))

# Compile the model
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# import libraries
import tensorflow as tf

# Create the model
model = tf.keras.Sequential()

# first convolutional layer
model.add(tf.keras.layers.Conv2D(filters = 6, kernel_size = (5, 5), activation = 'relu', input_shape = (28,28,1)))
# first pooling layer
model.add(tf.keras.layers.MaxPooling2D(pool_size = (2, 2)))
# second convolutional layer
model.add(tf.keras.layers.Conv2D(filters = 16, kernel_size = (5, 5), activation = 'relu'))
# second pooling layer
model.add(tf.keras.layers.MaxPooling2D(pool_size = (2, 2)))
# Flatten the output from the previous layers
model.add(tf.keras.layers.Flatten())
# first fully connected layer
model.add(tf.keras.layers.Dense(units = 120, activation = 'relu'))
# second fully connected layer
model.add(tf.keras.layers.Dense(units = 84, activation = 'relu'))

# output layer
model.add(tf.keras.layers.Dense(units = 10, activation = 'softmax'))

# Compile the model
model.compile(optimizer='adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# Train the model on dataset
model.fit(x_train, y_train, epochs = 10, batch_size = 32)

# Evaluate the model on test dataset
test_loss, test_acc = model.evaluate(x_test, y_test)

print("Test accuracy:", test_acc)

import tensorflow as tf

# Load the model from TensorFlow Hub
model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(None, None, 3)),
    tf.keras.applications.EfficientNetV2B1(weights = 'imagenet', include_top = True)
])

# Compile the model
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# Train the model on dataset
model.fit(X_train, y_train, epochs = 10, batch_size = 32)

# Evaluate the model on test dataset
test_loss, test_acc = model.evaluate(X_test, y_test)

print('Test accuracy:', test_acc)

import tensorflow as tf
# from tensorflow import keras
import keras
import loader
#全连接神经网络进行分类
data = loader.load()
(train_images, train_labels), (test_images, test_labels) = loader.make_dataset(data)
#ValueError: The first layer in a Sequential model must get an `input_shape` or `batch_input_shape` argument.
model = keras.Sequential([
    keras.layers.Flatten(),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dense(100, activation='relu'),
    keras.layers.Dense(2, activation=tf.nn.softmax)
])

model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])
model.fit(train_images, train_labels,
          batch_size=10,
          epochs=10,
          shuffle=True)
model.evaluate(test_images, test_labels)

# 0.8235
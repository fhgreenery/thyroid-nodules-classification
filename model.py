import tensorflow as tf
#使用keras框架搭建的模型,keras是tf的一个API，是tf的进一步封装，只能用python编程
import keras
import loader

data = loader.load()
(train_images, train_labels), (test_images, test_labels) = loader.make_dataset(data)

model = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3),
                        padding='SAME', input_shape=train_images.shape[1:], activation=tf.nn.relu),#32个3*3的filters 第一层需设置input_shape
    keras.layers.Conv2D(32, (3, 3), activation=tf.nn.relu),# train_images.shape:{430,128,128}
    keras.layers.MaxPool2D(pool_size=(2, 2)),
    keras.layers.Dropout(0.25),

    keras.layers.Conv2D(64, (3, 3), padding='SAME', activation=tf.nn.relu),
    keras.layers.Conv2D(64, (3, 3), activation=tf.nn.relu),
    keras.layers.MaxPool2D(pool_size=(2, 2)),
    keras.layers.Dropout(0.25),

    keras.layers.Flatten(),
    keras.layers.Dense(512, activation=tf.nn.relu),#512个神经元
    keras.layers.Dropout(0.5),
    keras.layers.Dense(2, activation=tf.nn.softmax)#全连接层 Softmax分类器输出的结果是输入样本在不同类别上的概率值大小
])

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.fit(train_images, train_labels,
          batch_size=10,
          epochs=10,
          shuffle=True)#返回train_loss train_acc
model.evaluate(test_images, test_labels)#返回test_loss test_acc

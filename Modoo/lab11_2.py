# tensorflow warning off
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# pooling layer

import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

image = tf.constant([[[[4], [3]],
                      [[2], [1]]]], dtype=np.float32)
pool = keras.layers.MaxPool2D(pool_size=(2, 2), strides=1, padding='VALID')(image)
print(image.shape)
print(pool.shape)
print(pool.numpy())
print("#################################")

# padding: same  --> output 3x3(input image와 동일)
image = tf.constant([[[[4], [3]],
                       [[2], [1]]]], dtype=np.float32)
pool = keras.layers.MaxPool2D(pool_size=(2, 2), strides=1, padding='SAME')(image)
print(image.shape)
print(pool.shape)
print(pool.numpy())
print("#################################")

mnist = keras.datasets.mnist
class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images.astype(np.float32) / 255.
test_images = test_images.astype(np.float32) / 255.

img = train_images[0]
plt.imshow(img, cmap='gray')
plt.show()

img = img.reshape(-1, 28, 28, 1)
img = tf.convert_to_tensor(img)
weight_init = keras.initializers.RandomNormal(stddev=0.01)
conv2d = keras.layers.Conv2D(filters=5, kernel_size=3, strides=(2, 2), padding='SAME',
                             kernel_initializer=weight_init)(img)
print(img.shape)
print(conv2d.shape)
feature_maps = np.swapaxes(conv2d, 0, 3)
for i, feature_map in enumerate(feature_maps):
    plt.subplot(1, 5, i+1), plt.imshow(feature_map.reshape(14, 14), cmap='gray')
plt.show()

pool = keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='SAME')(conv2d)
print(pool.shape)

feature_maps = np.swapaxes(pool, 0, 3)
for i, feature_map in enumerate(feature_maps):
    plt.subplot(1, 5, i+1), plt.imshow(feature_map.reshape(7, 7), cmap='gray')
plt.show()

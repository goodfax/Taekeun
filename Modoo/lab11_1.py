# tensorflow warning off
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# convolution layer
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

# toy data
image = tf.constant([[[[1], [2], [3]],
                      [[4], [5], [6]],
                      [[7], [8], [9]]]], dtype=np.float32)
print(image.shape)
plt.imshow(image.numpy().reshape(3, 3), cmap='Greys')
plt.show()

# simple convolution layer 1
# image(input): 1, 3, 3, 1 (batch, height, width, channel)
# weight shape(kernel_size): 2, 2, 1, 1 (height, width, in_channel, out_channel)
# conv shape(output): 1, 2, 2, 1 (batch, height, width, channel)
# stride: 1x1
# padding: valid  --> output 2x2
print("image.shape", image.shape)
weight = np.array([[[[1.]], [[1.]]],
                   [[[1.]], [[1.]]]])
print("weight.shape", weight.shape)
weight_init = tf.constant_initializer(weight)
conv2d = keras.layers.Conv2D(filters=1, kernel_size=2, padding='VALID',
                             kernel_initializer=weight_init)(image)
print("conv2d.shape", conv2d.shape)
print(conv2d.numpy().reshape(2, 2))
plt.imshow(conv2d.numpy().reshape(2, 2), cmap='gray')
plt.show()

# simple convolution layer 2
# image(input): 1, 3, 3, 1 (batch, height, width, channel)
# weight shape(kernel_size): 2, 2, 1, 1 (height, width, in_channel, out_channel)
# conv shape(output): 1, 3, 3, 1 (batch, height, width, channel)
# stride: 1x1
# padding: same  --> output 3x3(input image와 동일)

print("image.shape", image.shape)
weight = np.array([[[[1.]], [[1.]]],
                   [[[1.]], [[1.]]]])
print("weight.shape", weight.shape)
weight_init = tf.constant_initializer(weight)
conv2d = keras.layers.Conv2D(filters=1, kernel_size=2, padding='SAME',
                             kernel_initializer=weight_init)(image)
print("conv2d.shape", conv2d.shape)
print(conv2d.numpy().reshape(3, 3))
plt.imshow(conv2d.numpy().reshape(3, 3), cmap='gray')
plt.show()

# simple convolution layer 3
# image(input): 1, 3, 3, 1 (batch, height, width, channel)
# weight shape(kernel_size): 2, 2, 1, 3 (height, width, in_channel, out_channel)
# conv shape(output): 1, 3, 3, 1 (batch, height, width, channel)
# stride: 1x1
# padding: same  --> output 3x3(input image와 동일)
print("image.shape", image.shape)
weight = np.array([[[[1., 10., -1.]], [[1., 10., -1.]]],
                   [[[1., 10., -1.]], [[1., 10., -1.]]]])
print("weight.shape", weight.shape)
weight_init = tf.constant_initializer(weight)
donv2d = keras.layers.Conv2D(filters=3, kernel_size=2, padding='SAME',
                             kernel_initializer=weight_init)(image)
print("conv2d.shape", conv2d.shape)
feature_map = np.swapaxes(conv2d, 0, 3)
for i, feature_map in enumerate(feature_map):
    print(feature_map.reshape(3, 3))
    plt.subplot(1, 3, i+1), plt.imshow(feature_map.reshape(3, 3), cmap='gray')
plt.show()


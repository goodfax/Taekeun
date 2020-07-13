# lab09_1.py
# Logistic Regression으로 XOR 문제 풀어보기
# 당연히 안풀린다.

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf

tf.random.set_seed(777)

# generated XOR data set
x_data = [[0, 0],
          [0, 1],
          [1, 0],
          [1, 1]]
y_data = [[0],
          [1],
          [1],
          [0]]

plt.scatter(x_data[0][0], x_data[0][1], c='red', marker='^')
plt.scatter(x_data[3][0], x_data[3][1], c='red', marker='^')
plt.scatter(x_data[1][0], x_data[1][1], c='blue', marker='^')
plt.scatter(x_data[2][0], x_data[2][1], c='blue', marker='^')

plt.xlabel("x1")
plt.ylabel("x2")
plt.show()

# data preprocessing
dataset = tf.data.Dataset.from_tensor_slices((x_data, y_data)).batch(len(x_data))

def preprocess_data(features, labels):
    features = tf.cast(features, tf.float32)
    labels = tf.cast(labels, tf.float32)
    return features, labels

# initialization of Weight and bias
W = tf.Variable(tf.zeros((2, 1)), name='weight')
b = tf.Variable(tf.zeros((1,)), name='bias')
print("W = {}, B = {}".format(W.numpy(), b.numpy()))

# hypothesis define(sigmoid)
def logistic_regression(features):
    hypothesis = tf.divide(1., 1. + tf.exp(tf.matmul(features, W) + b))
    return hypothesis

# Cost function define
def loss_fn(hypothesis, features, labels):
    cost = -tf.reduce_mean(labels * tf.math.log(logistic_regression(features))
                           + (1 - labels) * tf.math.log(1 - hypothesis))
    return cost

# optimizer define
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

# accuracy
# sigmoid 값(hypothesis 값)을 통해 0.5보다 크면 1을 반환하고 0.5보다 작으면 0을 반환
def accuracy_fn(hypothesis, labels):
    predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, labels), dtype=tf.float32))
    return accuracy

# gradient descent
def grad(hypothesis, features, labels):
    with tf.GradientTape() as tape:
        loss_value = loss_fn(logistic_regression(features), features, labels)
    return tape.gradient(loss_value, [W, b])

EPOCHS = 1001
for step in range(EPOCHS):
    for features, labels in dataset:
        features, labels = preprocess_data(features, labels)
        grads = grad(logistic_regression(features), features, labels)
        optimizer.apply_gradients(grads_and_vars=zip(grads, [W,b]))
        if step % 100 == 0:
            print("Step: {}, Loss: {:.4f}".format(step,
                                                  loss_fn(logistic_regression(features),
                                                          features, labels)))

print("W = {}, b = {}".format(W.numpy(), b.numpy()))
x_data, y_data = preprocess_data(x_data, y_data)
test_acc = accuracy_fn(logistic_regression(x_data), y_data)
print("Testing Accuracy: {:.4f}".format(test_acc))

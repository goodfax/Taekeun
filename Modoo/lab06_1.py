import numpy as np
import tensorflow as tf
from pprint import pprint

# Sample dataset
x_data = [[1, 2, 1, 1],
          [2, 1, 3, 2],
          [3, 1, 3, 4],
          [4, 1, 5, 5],
          [1, 7, 5, 5],
          [1, 2, 5, 6],
          [1, 6, 6, 6],
          [1, 7, 7, 7]
          ]

y_data = [[0, 0, 1],
          [0, 0, 1],
          [0, 0, 1],
          [0, 1, 0],
          [0, 1, 0],
          [0, 1, 0],
          [1, 0, 0],
          [1, 0, 0]
          ]

x_data = np.asarray(x_data, dtype=np.float32)
y_data = np.asarray(y_data, dtype=np.float32)

nb_classes = 3

# Weight and bias
W = tf.Variable(tf.random.normal([4, nb_classes]), name='weight')
b = tf.Variable(tf.random.normal([nb_classes]), name='bias')
variables = [W, b]
print('##Weight and bias##')
pprint(variables)

# softmax function example
sample_db = [[8, 2, 1, 4]]
sample_db = np.asarray(sample_db, dtype=np.float32)
print('##sample db##')
print(sample_db)
sample_hypothesis = tf.nn.softmax(tf.matmul(sample_db, W) + b)
print('##softmax function example##')
pprint(sample_hypothesis)

# cost function
def cost_fn(X, Y):
    logits = tf.nn.softmax(tf.matmul(X, W) + b)
    cost = -tf.reduce_sum(Y * tf.math.log(logits), axis=1)
    cost_mean = tf.reduce_mean(cost)
    return cost_mean
print('##cost function##')
pprint(cost_fn(x_data, y_data))

# gradient descent function
def grad_fn(X, Y):
    with tf.GradientTape() as tape:
        cost = cost_fn(X, Y)
        grads = tape.gradient(cost, variables)
        return grads
print('##gradient descent function##')
pprint(grad_fn(x_data, y_data))

# training function
def fit(X, Y, epochs=2000, verbose=100):
    optimizer = tf.optimizers.SGD(learning_rate=0.1)
    for i in range(epochs):
        grads = grad_fn(X, Y)
        optimizer.apply_gradients(zip(grads, variables))
        if (i == 0) | ((i+1)%verbose == 0):
            print('Loss at epoch %d: %f' %(i+1, cost_fn(X, Y).numpy()))


print('##training function##')
fit(x_data, y_data)

# prediction
a = tf.nn.softmax(tf.matmul(x_data, W) + b)
print('##prediction##')
pprint(a)
pprint(tf.argmax(a, 1))
pprint(tf.argmax(y_data, 1))
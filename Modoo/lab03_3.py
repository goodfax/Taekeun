# lab03_3.py
import tensorflow as tf

# tensorflow 1.x
# tf.set_random_seed(0)

# tensorflow 2.0
tf.random.set_seed(0)

x_data = [1., 2., 3., 4.]
y_data = [1., 3., 5., 7.]

# tensorflow 1.x
# W = tv.Variable(tf.random_normal([1], -100., 100.))
# tensorflow 2.0
W = tf.Variable(tf.random.normal([1], -100., 100.))

for step in range(300):
    hypothesis = W * x_data
    cost = tf.reduce_mean(tf.square(hypothesis - y_data))

    alpha = 0.01
    gradient = tf.reduce_mean(tf.multiply(tf.multiply(W, x_data) - y_data, x_data))
    descent = W - tf.multiply(alpha, gradient)
    W.assign(descent)

    if step % 10 == 0:
        print('{:5} | {:10.4f} | {:10.6f}'.format(step, cost.numpy(), W.numpy()[0]))
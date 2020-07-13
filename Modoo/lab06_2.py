import numpy as np
import tensorflow as tf

xy = np.loadtxt('data-04-zoo.csv', delimiter=',', dtype=np.float32)
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]

print(x_data.shape, y_data.shape)

nb_classes = 7
# list(y_data) -> tf.cast(list(y_data))
Y_one_hot = tf.one_hot(tf.cast(list(y_data), tf.int32), nb_classes)
Y_one_hot = tf.reshape(Y_one_hot, [-1, nb_classes])

W = tf.Variable(tf.random.normal([16, nb_classes]), name='weight')
b = tf.Variable(tf.random.normal([nb_classes]), name='bias')
variables = [W, b]

def logit_fn(X):
    return tf.matmul(X, W) + b

def hypothesis(X):
    return tf.nn.softmax(logit_fn(X))

def cost_fn(X, Y):
    logits = logit_fn(X)
    cost_i = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y)

    cost = tf.reduce_mean(cost_i)
    return cost

def grad_fn(X, Y):
    with tf.GradientTape() as tape:
        loss = cost_fn(X, Y)
        grads = tape.gradient(loss, variables)
        return grads

def prediction(X, Y):
    pred = tf.argmax(hypothesis(X), 1)
    correct_prediction = tf.equal(pred, tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return accuracy

def fit(X, Y, epochs=500, verbose=50):
    optimizer = tf.optimizers.SGD(learning_rate=0.1)

    for i in range(epochs):
        grads = grad_fn(X, Y)
        optimizer.apply_gradients(zip(grads, variables))
        if (i == 0) | ((i+1) % verbose == 0):
            acc = prediction(X, Y).numpy()
            loss = tf.reduce_sum(cost_fn(X, Y)).numpy()

            print('Loss & Acc at {} epoch {}, {}'.format(i+1, loss, acc))

fit(x_data, Y_one_hot, epochs=1101)

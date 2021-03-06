import tensorflow as tf

x_train = [[1., 2.],
           [2., 3.],
           [3., 1.],
           [4., 3.],
           [5., 3.],
           [6., 2.]
]

y_train = [[0.],
           [0.],
           [0.],
           [1.],
           [1.],
           [1.]
]

x_test = [[5., 2.]]
y_test = [[1.]]

dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(len(x_train))
W = tf.Variable(tf.zeros([2, 1]), name='weight')
b = tf.Variable(tf.zeros([1]), name='bias')

def logistic_regression(features):
    # tensorflow 1.x
    # tf.div()

    # tensorflow 2.x
    hypothesis = tf.math.divide(1., 1. + tf.exp(tf.matmul(features, W) + b))
    return hypothesis

def loss_fn(features, labels):
    hypothesis = logistic_regression(features)
    # tensorflow 1.x
    # tf.log()

    # tensorflow 2.x
    cost = - tf.reduce_mean(labels * tf.math.log(hypothesis)
                            + (1 - labels) * tf.math.log(1 - hypothesis))
    return cost

def grad(features, labels):
    with tf.GradientTape() as tape:
        loss_value = loss_fn(features, labels)
    return tape.gradient(loss_value, [W, b])

def accuracy_fn(hypothesis, labels):
    predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, labels), dtype=tf.int32))
    return accuracy

# tensorflow 1.x
# tf.train.GradientDescentOptimizer()

# tensorflow 2.x
optimizer = tf.optimizers.SGD(learning_rate=0.01)
EPOCHS = 1001

for step in range(EPOCHS):
    for features, labels in tf.data.Dataset.as_numpy_iterator(dataset):
        grads = grad(features, labels)
        optimizer.apply_gradients(grads_and_vars=zip(grads, [W, b]))
        if step % 100 == 0:
            print("Iter: {}, Loss: {:.4f}".format(step, loss_fn(features, labels)))

test_acc = accuracy_fn(logistic_regression(x_test), y_test)
print("{:.4f}".format(test_acc))


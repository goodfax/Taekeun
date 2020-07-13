# CIFAR-100 분류해보기

import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras

tf.random.set_seed(777)

# train data = 50000
# test data = 10000
cifar = tf.keras.datasets.cifar100
(train_images, train_labels), (test_images, test_labels) = cifar.load_data(
    label_mode='coarse')

# 20개의 label
class_names = ['aquatic mammals', 'fish', 'flowers', 'food containers',
               'fruit/vegetables', 'household electrical devices',
               'household furniture', 'insects', 'large carnivores',
               'large man-made outdoor things', 'large natural outdoor scenes',
               'large omnivores/herbivores', 'medium-sized mammals',
               'non-insect invertebrates', 'people', 'reptiles', 'small mammals',
               'trees', 'vehicles 1', 'vehicles 2']

# data 하나 뽑아서 보기
plt.figure()
plt.imshow(train_images[3])
plt.colorbar()
plt.grid(False)
plt.show()

# 0~1 값으로 정규화
train_images = train_images / 255.0
test_images = test_images / 255.0

# 정규화한 데이터 출력으로 확인
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[int(train_labels[i])])
plt.show()

# Tensorflow keras로 모델 정의
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(32, 32, 3)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(20, activation=tf.nn.softmax)
])

model.summary()

# train data = 40000
# validation data = 10000
# test data = 10000
x_val = train_images[:10000]
partial_x_train = train_images[10000:]

y_val = train_labels[:10000]
partial_y_train = train_labels[10000:]

# optimizer로 ADAM 사용
# Epoch은 100
# verbose=2: 막대바 X
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(partial_x_train,
          partial_y_train,
          validation_data=(x_val, y_val),
          epochs=100,
          verbose=2)

# verbose=2: 막대바 X
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print('Test accuracy:', test_acc)
print('Test loss', test_loss)

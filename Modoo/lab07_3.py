# IMDB 긍부정 분류해보기

import numpy as np
import tensorflow as tf
from tensorflow import keras

tf.random.set_seed(777)

# imdb dataset load
# 빈도수가 높은 상위 10000개의 word vector를 사용
imdb = keras.datasets.imdb
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
print("training entries: {}, labels: {}".format(len(train_data), len(train_labels)))
print(train_data[0])

# word vector to real word
word_index = imdb.get_word_index()

word_index = {k:(v+3) for k,v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def decode_review(test):
    return ' '.join([reverse_word_index.get(i, '?') for i in test])

# 4번째 데이터에 대한 test/label 출력
print(decode_review(train_data[4]))
print(train_labels[4])

# train/test data preprocessing(word embedding)
train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                                                        value=word_index["<PAD>"],
                                                        padding='post',
                                                        maxlen=256)
test_data = keras.preprocessing.sequence.pad_sequences(test_data,
                                                       value=word_index["<PAD>"],
                                                       padding='post',
                                                       maxlen=256)
print(len(train_data[0]), len(test_data[0]))
print(train_data[0])
print(decode_review(train_data[0]))

# Tensorflow keras로 모델 정의
vocab_size = 10000

model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation=tf.nn.relu))
model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))

# model summary
model.summary()

# optimizer로 ADAM 사용
# Epoch은 40
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy']
              )
x_val = train_data[:10000]
partial_x_train = train_data[10000:]

y_val = train_labels[:10000]
partial_y_train = train_labels[10000:]

# verbose=2: 막대바X
model.fit(partial_x_train,
          partial_y_train,
          epochs=40,
          batch_size=512,
          validation_data=(x_val, y_val),
          verbose=2)

# verbose=2: 막대바X
test_loss, test_acc = model.evaluate(test_data, test_labels, verbose=2)
print('Test accuracy:', test_acc)
print('Test loss', test_loss)
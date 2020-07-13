# RNN으로 many to one 문제 풀기
# word sentiment classification

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Sequential, Model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# prepairing dataset
# example data
words = ['good', 'bad', 'worse', 'so good']
y_data = [1, 0, 0, 1]

# dictionary 생성
char_set = ['<pad>'] + sorted(list(set(''.join(words))))
idx2char = {idx: char for idx, char in enumerate(char_set)}
char2idx = {char: idx for idx, char in enumerate(char_set)}
print('# dictionary 생성 #')
print(char_set)
print(idx2char)
print(char2idx)

x_data = list(map(lambda word: [char2idx.get(char) for char in word], words))
x_data_len = list(map(lambda word: len(word), x_data))
print('# token sequence를 index sequence로 변환(char -> dictionary index of char) #')
print(x_data)
print(x_data_len)

# padding
max_sequence = 10
x_data = pad_sequences(sequences=x_data, maxlen=max_sequence, padding='post', truncating='post')
print('# padding한 데이터 #')
print(x_data)
print(x_data_len)
print(y_data)

# 모델 생성
input_dim = len(char2idx)
output_dim = len(char2idx)
one_hot = np.eye(len(char2idx))
hidden_size = 10
num_classes = 2

model = Sequential()
model.add(layers.Embedding(input_dim=input_dim, output_dim=output_dim, trainable=False, mask_zero=True,
                           input_length=max_sequence, embeddings_initializer=keras.initializers.Constant(one_hot)))
model.add(layers.SimpleRNN(units=hidden_size))
model.add(layers.Dense(units=num_classes))
model.summary()

# 모델 학습
def loss_fn(model, x, y):
    return tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(y_true=y, y_pred=model(x), from_logits=True))

# hyper parameter setting
lr = .01
epochs = 30
batch_size = 2
opt = tf.keras.optimizers.Adam(learning_rate=lr)

# data pipeline 생성
tr_dataset = tf.data.Dataset.from_tensor_slices((x_data, y_data))
tr_dataset = tr_dataset.shuffle(buffer_size=4)
tr_dataset = tr_dataset.batch(batch_size=batch_size)
print('# input dataset information #')
print(tr_dataset)

# 학습
tr_loss_hist = []

for epoch in range(epochs):
    avg_tr_loss = 0
    tr_step = 0

    for x_mb, y_mb in tr_dataset:
        with tf.GradientTape() as tape:
            tr_loss = loss_fn(model, x=x_mb, y=y_mb)
        grads = tape.gradient(target=tr_loss, sources=model.variables)
        opt.apply_gradients(grads_and_vars=zip(grads, model.variables))
        avg_tr_loss += tr_loss
        tr_step += 1
    else:
        avg_tr_loss /= tr_step
        tr_loss_hist.append(avg_tr_loss)

    if (epoch + 1) % 5 == 0:
        print('epoch : {:3}, tr_loss : {:.3f}'.format(epoch + 1, avg_tr_loss.numpy()))

# checking performance
yhat = model.predict(x_data)
yhat = np.argmax(yhat, axis=-1)
print('acc : {:.2%}'.format(np.mean(yhat == y_data)))

plt.plot(tr_loss_hist)
plt.show()

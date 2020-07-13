# rnn basic
import numpy as np
from tensorflow.keras import layers

# data set
h = [1, 0, 0, 0]
e = [0, 1, 0, 0]
l = [0, 0, 1, 0]
o = [0, 0, 0, 1]

# One cell: input_dim=4, hidden_size(output_dimension)=2
x_data = np.array([[h]], dtype=np.float32)

hidden_size = 2
cell = layers.SimpleRNNCell(units=hidden_size)
rnn = layers.RNN(cell, return_sequences=True, return_state=True)
outputs, states = rnn(x_data)

print('# One cell: input_dim=4, hidden_size=2 #')
print('x_data: {}, shape: {}'.format(x_data, x_data.shape))
print('outputs: {}, shape: {}'.format(outputs, outputs.shape))
print('states: {}, shape: {}'.format(states, states.shape))

# SimpleRNNCell + RNN(?)
rnn = layers.SimpleRNN(units=hidden_size, return_sequences=True, return_state=True)
outputs, states = rnn(x_data)

print('# SimpleRNNCell + RNN(?) #')
print('x_data: {}, shape: {}'.format(x_data, x_data.shape))
print('outputs: {}, shape: {}'.format(outputs, outputs.shape))
print('states: {}, shape: {}'.format(states, states.shape))

# 여러개 sequence를 갖는 RNN
# One cell: input_dim=4, hidden_size(output_dimension)=2, sequence=5
x_data = np.array([[h, e, l, l, o]], dtype=np.float32)

hidden_size = 2
rnn = layers.SimpleRNN(units=hidden_size, return_sequences=True, return_state=True)
outputs, states = rnn(x_data)

print('# One cell: input_dim=4, hidden_size=2, sequence=5 #')
print('x_data: {}, shape: {}'.format(x_data, x_data.shape))
print('outputs: {}, shape: {},'.format(outputs, outputs.shape))
print('states: {}, shape: {}'.format(states, states.shape))

# Batching Input
# One cell: input_dim=4, hidden_size(output_dimension)=2, sequence=5, batch=3
# 3 batch = 'hello', 'eolll', 'lleel'
x_data = np.array([[h, e, l, l, o],
                   [e, o, l, l, l],
                   [l, l, e, e, l]], dtype=np.float32)
hidden_size = 2
rnn = layers.SimpleRNN(units=hidden_size, return_sequences=True, return_state=True)
outputs, states = rnn(x_data)

print('# One cell: input_dim=4, hidden_size(output_dimension)=2, sequence=5, batch=3 #')
print('x_data: {}, shape: {}'.format(x_data, x_data.shape))
print('outputs: {}, shape: {}'.format(outputs, outputs.shape))
print('states: {}, shape: {}'.format(states, states.shape))

import os
import sys
import string
import numpy as np
from numpy.lib.arraypad import pad
import pandas as pd
import matplotlib.pyplot as plt

from keras.models import Model
from keras.layers import Dense, Embedding, Input, LSTM
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.optimizers import Adam, SGD
from tensorflow.python.keras.preprocessing.text import one_hot
from tensorflow.python.ops.gen_math_ops import Mod

try:
  import keras.backend as K
  if len(K.tensorflow_backend._get_available_gpus()) > 0:
    from keras.layers import CuDNNLSTM as LSTM
    from keras.layers import CuDNNGRU as GRU
except:
  pass


MAX_SEQUENCE_LENGTH = 100
MAX_VOCAB_SIZE = 3000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2
BATCH_SIZE = 256
EPOCHS = 2000
LATENT_DIM = 50


input_texts = []
output_texts = []
for line in open('nlp/robert_frost.txt'):
    line = line.rstrip()
    if not line:
        continue
    
    input_line = '<sos> ' + line
    output_line = line + ' <eos>'
    input_texts.append(input_line)
    output_texts.append(output_line)
    
all_lines = input_texts + output_texts


tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE, filters='')
tokenizer.fit_on_texts(all_lines)
input_sequences = tokenizer.texts_to_sequences(input_texts)
output_sequences = tokenizer.texts_to_sequences(output_texts)

max_sequence_length_from_data = max(len(s) for s in input_sequences)
print("Max Sequence Length: {}".format(max_sequence_length_from_data))


word2idx = tokenizer.word_index
print("Found {} unique tokens".format(len(word2idx)))
# assert('<sos>' in word2idx)
# assert('<eos>' in word2idx)


max_sequence_length = min(max_sequence_length_from_data, MAX_SEQUENCE_LENGTH)
input_sequences = pad_sequences(input_sequences, maxlen=max_sequence_length, padding='post', truncating='post')
output_sequences = pad_sequences(output_sequences, maxlen=max_sequence_length, padding='post',  truncating='post')


print("Loading Word Vectors...")
word2vec = {}
with open('nlp/glove.6B/glove.6B.{}d.txt'.format(EMBEDDING_DIM), encoding='utf8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        vectors = np.array(values[1:])
        word2vec[word] = vectors
print("Found {} of Word Vectors".format(len(word2vec)))


print("Filling pretrained Embeddings")
num_words = min(MAX_VOCAB_SIZE, len(word2idx) + 1)
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
for word, i in word2idx.items():
    if i < MAX_VOCAB_SIZE:
        embedding_vector = word2vec.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

one_hot_targets = np.zeros((len(input_sequences), max_sequence_length, num_words))
for i, output_sequence in enumerate(output_sequences):
    for t, word in enumerate(output_sequence):
        if word > 0:
            one_hot_targets[i, t, word] = 1
            
embedding_layer = Embedding(num_words, 
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            )


print("Building Model...")
input_ = Input(shape=(max_sequence_length,))
initial_h = Input(shape=(LATENT_DIM,))
initial_c = Input(shape=(LATENT_DIM,))
x = embedding_layer(input_)
lstm = LSTM(LATENT_DIM, return_sequences=True, return_state=True)
x, _, _ = lstm(x, initial_state=[initial_h, initial_c])
dense = Dense(num_words, activation='softmax')
output = dense(x)

model = Model(inputs=[input_, initial_h, initial_c], outputs=output)
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.01), metrics=['accuracy'])

print("Training Model")
z = np.zeros((len(input_sequences), LATENT_DIM))
r = model.fit(
    [input_sequences, z, z],
    one_hot_targets,
    batch_size = BATCH_SIZE,
    epochs= EPOCHS,
    validation_split= VALIDATION_SPLIT)


# plot some loss
plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['val_loss'], label='val_loss')
plt.legend()
plt.show()

# accuracies
plt.plot(r.history['accuracy'], label='acc')
plt.plot(r.history['val_accuracy'], label='val_acc')
plt.legend()
plt.show()

# make a sampling model
input2 = Input(shape=(1,))
x = embedding_layer(input2)
x, h, c = lstm(x, initial_state=[initial_h, initial_c])
output2 = dense(x)
sampling_model = Model(inputs=[input2, initial_h, initial_c], outputs=[output2, h, c])


# reverse word2idx dictionary to get back words
# during prediction
idx2word = {v:k for k,v in word2idx.items()}


def sample_line():
    np_input = np.array([[ word2idx['<sos>'] ]])
    h = np.zeros((1, LATENT_DIM))
    c = np.zeros((1, LATENT_DIM))
    
    eos = word2idx['<eos>']
    output_sentence = []
    for _ in range(max_sequence_length):
        o, h, c = sampling_model.predict([np_input, h, c])
        probs = o[0,0]
        if np.argmax(probs) == 0:
            print("WTF")
        probs[0] = 0
        probs /= probs.sum()
        idx = np.random.choice(len(probs), p=probs)
        if idx == eos:
            break
        
        output_sentence.append(idx2word.get(idx, '<WTF {}>'.format(idx)))
        # make the next input into model
        np_input[0,0] = idx
    return ' '.join(output_sentence)

while True:
    for _ in range(5):
        created_line = sample_line()
        print(created_line)
        with open('nlp/poetry_generated.txt', "a") as f:
            f.write(created_line + '\n')
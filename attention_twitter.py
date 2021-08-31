from operator import le
import os, sys
import re
from re import I, S 
from keras.models import Model
from keras.layers import Bidirectional, LSTM, Input, GRU, Dense, Embedding, RepeatVector, \
                         Concatenate, Activation, Dot, Lambda
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import keras.backend as K

import numpy as np 
import matplotlib.pyplot as plt
from numpy.lib.arraypad import pad 
import pandas as pd
from tensorflow.python.keras.backend import shape, stack 

try:
  if len(K.tensorflow_backend._get_available_gpus()) > 0:
    from keras.layers import CuDNNLSTM as LSTM
    from keras.layers import CuDNNGRU as GRU
except:
  pass


def softmax_over_time(x):
    assert(K.ndim(x) > 2)
    e = K.exp(x - K.max(x, axis=1, keepdims=True))
    s = K.sum(e, axis=1, keepdims=True)
    return e/s 


# Config
BATCH_SIZE = 32
EPOCHS = 30 
LATENT_DIM = 512
LATENT_DIM_DECODER = 400
NUM_SAMPLES = 20000
MAX_SEQUENCE_LENGTH = 100
MAX_NUM_WORDS = 20000
EMBEDDING_DIM = 200


# TEXT
input_texts = []
target_texts = []
target_texts_inputs = []

t = 0
for line in open('nlp/twitter_tab_format.txt', encoding='UTF-8'):
    t += 1 
    if t > NUM_SAMPLES:
        break
    if '\t' not in line:
        continue
    
    input_text, translation, *rest = line.rstrip().split('\t')
    
    target_texts_input = '<sos> ' + translation
    target_text = translation + ' <eos>'
    
    input_texts.append(input_text)
    target_texts_inputs.append(target_texts_input)    
    target_texts.append(target_text)
print("Num Samples: ", len(input_text))


# Tokenize Input
tokenizer_input = Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer_input.fit_on_texts(input_texts)
input_sequences = tokenizer_input.texts_to_sequences(input_texts)

# Word2idx, max_len
word2idx_inputs = tokenizer_input.word_index
print("Found {} unique input tokens.".format(len(word2idx_inputs)))
max_len_inputs = max([len(s) for s in input_sequences])


# Tokenize Output
tokenizer_output = Tokenizer(num_words=MAX_NUM_WORDS, filters=' ')
tokenizer_output.fit_on_texts(target_texts + target_texts_inputs)
target_sequences = tokenizer_output.texts_to_sequences(target_texts)
target_sequences_inputs = tokenizer_output.texts_to_sequences(target_texts_inputs)

# Word2idx, max_len
word2idx_outputs = tokenizer_output.word_index
print("Found {} unique input tokens.".format(len(word2idx_outputs)))
max_len_outputs = max([len(s) for s in target_sequences])
num_words_output = len(word2idx_outputs) + 1


# Padding Sequences 
encoder_inputs = pad_sequences(input_sequences, padding='pre', maxlen=max_len_inputs)
print("encoder_data.shape:", encoder_inputs.shape)
print("encoder_data[0]:", encoder_inputs[0])

decoder_inputs = pad_sequences(target_sequences_inputs, padding='post', maxlen=max_len_outputs)
decoder_targets = pad_sequences(target_sequences, maxlen=max_len_outputs, padding='post')




# Pre-trained Word Vectors
print('Filling Word Vectos...')
word2vec = {}
with open('nlp/glove.6B/glove.6B.{}d.txt'.format(EMBEDDING_DIM), encoding='UTF-8') as F:
    for line in F:
        values = line.split()
        word = values[0]
        vec = np.asarray(values[1:])
        word2vec[word] = vec
print("Found {} word vectors".format(len(word2vec)))     


# Embedding Matrix
print('Filling pre-trained embeddings...')
num_words = min(MAX_NUM_WORDS, len(word2idx_inputs) + 1)
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
for word, i in word2idx_inputs.items():
    if i < MAX_NUM_WORDS:
        embedding_vector = word2vec.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector


# Create Embedding Layer
embedding_layer = Embedding(num_words, 
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=max_len_inputs,
                            trainable=False
                            )

decoder_targets_one_hot = np.zeros(
    (
        len(input_texts),
        max_len_outputs,
        num_words_output
    )
)

for i, d in enumerate(decoder_targets):
    for t, word in enumerate(d):
        if word > 0:
            decoder_targets_one_hot[i, t, word] = 1


# Building Model
print("Building the Model")
# Encoder
encoder_inputs_placeholder = Input(shape=(max_len_inputs,))
x = embedding_layer(encoder_inputs_placeholder)
encoder = Bidirectional(LSTM(LATENT_DIM, return_sequences=True, dropout=0.5))
encoder_outputs = encoder(x)


# Decoder
decoder_inputs_placeholder = Input(shape=(max_len_outputs,))
decoder_embedding = Embedding(num_words_output, EMBEDDING_DIM)
decoder_inputs_x = decoder_embedding(decoder_inputs_placeholder)


######### Attention #########
# Attention layers need to be global because
# they will be repeated Ty times at the decoder

attn_repeat_layer = RepeatVector(max_len_inputs)
attn_concat_layer = Concatenate(axis=-1)
attn_dense_1 = Dense(10, activation='tanh')
attn_dense_2 = Dense(1, activation=softmax_over_time)
attn_dot = Dot(axes=1) # to perform the weighted sum of alpha[t] * h[t]


def one_step_attention(h, st_1):
    
    
    # h = h(1), ..., h(Tx), shape = (Tx, LATENT_DIM * 2)
    # st_1 = s(t-1), shape = (LATENT_DIM_DECODER,)
      
    # copy s(t-1) Tx times
    # now shape = (Tx, LATENT_DIM_DECODER)
    st_1 = attn_repeat_layer(st_1)
    
    # Concatenate all h(t)'s with s(t-1)
    # Now of shape (Tx, LATENT_DIM_DECODER + LATENT_DIM * 2)  
    x = attn_concat_layer([h, st_1])
    x = attn_dense_1(x)
    alphas = attn_dense_2(x)
    # "Dot" the alphas and the h's
    # Remember a.dot(b) = sum over a[t] * b[t]
    context = attn_dot([alphas, h])
    return context


# After Attention Decoder 
decoder_lstm = LSTM(LATENT_DIM_DECODER, return_state= True)
decoder_dense = Dense(num_words_output, activation='softmax')

initial_s = Input(shape=(LATENT_DIM_DECODER, ), name='s0')
initial_c = Input(shape=(LATENT_DIM_DECODER, ), name='c0')
context_last_word_concat_layer = Concatenate(axis=2)

s = initial_s
c = initial_c

outputs = []
for t in range(max_len_outputs):
    context = one_step_attention(encoder_outputs, s)
    
    # we need a different layer for each time step    
    selector = Lambda(lambda x : x[:, t:t+1])
    xt = selector(decoder_inputs_x)
    
    # Combine
    decoder_lstm_input = context_last_word_concat_layer([context, xt])
    # pass the combined [context, last word] into the LSTM
    # along with [s, c]
    # get the new [s, c] and output    
    o, s, c = decoder_lstm(decoder_lstm_input, initial_state=[s,c])
    
    decoder_outputs = decoder_dense(o)
    outputs.append(decoder_outputs)


def stack_and_transpose(x):
    # x is a list of length T, each element is a batch_size x output_vocab_size tensor
    x = K.stack(x) # is now T x batch_size x output_vocab_size tensor
    x = K.permute_dimensions(x, pattern=(1, 0, 2)) # is now batch_size x T x output_vocab_size
    return x   


stacker = Lambda(stack_and_transpose) 
outputs = stacker(outputs)

model = Model(inputs=[encoder_inputs_placeholder,
                      decoder_inputs_placeholder,
                      initial_s,
                      initial_c],
              outputs=outputs)

def custom_loss(y_true, y_pred):
      # both are of shape N x T x K
  mask = K.cast(y_true > 0, dtype='float32')
  out = mask * y_true * K.log(y_pred)
  return -K.sum(out) / K.sum(mask)


def acc(y_true, y_pred):
  # both are of shape N x T x K
  targ = K.argmax(y_true, axis=-1)
  pred = K.argmax(y_pred, axis=-1)
  correct = K.cast(K.equal(targ, pred), dtype='float32')

  # 0 is padding, don't include those
  mask = K.cast(K.greater(targ, 0), dtype='float32')
  n_correct = K.sum(mask * correct)
  n_total = K.sum(mask)
  return n_correct / n_total


model.compile(loss=custom_loss, optimizer='adam', metrics=['acc'])
model.summary()

z = np.zeros((len(encoder_inputs), LATENT_DIM_DECODER))
r = model.fit(
    [encoder_inputs, decoder_inputs, z, z],
    decoder_targets_one_hot,
    batch_size = BATCH_SIZE,
    epochs = EPOCHS,
    validation_split= 0.2
)




# plot some data
plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['val_loss'], label='val_loss')
plt.legend()
plt.show()

# accuracies
plt.plot(r.history['acc'], label='acc')
plt.plot(r.history['val_acc'], label='val_acc')
plt.legend()
plt.show()






##### Make predictions #####
# As with the poetry example, we need to create another model
# that can take in the RNN state and previous word as input
# and accept a T=1 sequence.

# The encoder will be stand-alone
# From this we will get our initial decoder hidden state
# i.e. h(1), ..., h(Tx)
encoder_model = Model(inputs=encoder_inputs_placeholder, outputs=encoder_outputs)

# next we define a T=1 decoder model
encoder_outputs_as_input = Input(shape=(max_len_inputs, LATENT_DIM * 2,))
decoder_inputs_single = Input(shape=(1,))
decoder_inputs_single_x = decoder_embedding(decoder_inputs_single)

# no need to loop over attention steps this time because there is only one step
context = one_step_attention(encoder_outputs_as_input, initial_s)
# combine context with last word
decoder_lstm_input = context_last_word_concat_layer([context,  decoder_inputs_single_x])
o, s, c = decoder_lstm(decoder_lstm_input, initial_state=[initial_s, initial_c])
decoder_outputs = decoder_dense(o)

# note: we don't really need the final stack and tranpose
# because there's only 1 output
# it is already of size N x D
# no need to make it 1 x N x D --> N x 1 x D
decoder_model = Model(
  inputs=[
    decoder_inputs_single,
    encoder_outputs_as_input,
    initial_s, 
    initial_c
  ],
  outputs=[decoder_outputs, s, c]
)


# map indexes back into real words
# so we can view the results
idx2word_eng = {v:k for k, v in word2idx_inputs.items()}
idx2word_trans = {v:k for k, v in word2idx_outputs.items()}


def decode_sequence(input_seq):
      # Encode the input as state vectors.
  enc_out = encoder_model.predict(input_seq)

  # Generate empty target sequence of length 1.
  target_seq = np.zeros((1, 1))
  
  # Populate the first character of target sequence with the start character.
  # NOTE: tokenizer lower-cases all words
  target_seq[0, 0] = word2idx_outputs['<sos>']

  # if we get this we break
  eos = word2idx_outputs['<eos>']


  # [s, c] will be updated in each loop iteration
  s = np.zeros((1, LATENT_DIM_DECODER))
  c = np.zeros((1, LATENT_DIM_DECODER))


  # Create the translation
  output_sentence = []
  for _ in range(max_len_outputs):
    o, s, c = decoder_model.predict([target_seq, enc_out, s, c])
        

    # Get next word
    idx = np.argmax(o.flatten())

    # End sentence of EOS
    if eos == idx:
      break

    word = ''
    if idx > 0:
      word = idx2word_trans[idx]
      output_sentence.append(word)

    # Update the decoder input
    # which is just the word just generated
    target_seq[0, 0] = idx

  return ' '.join(output_sentence)




while True:
  # Do some test translations
  i = np.random.choice(len(input_texts))
  input_seq = encoder_inputs[i:i+1]
  translation = decode_sequence(input_seq)
  print('-')
  print('Input sentence:', input_texts[i])
  print('Predicted translation:', translation)
  print('Actual translation:', target_texts[i])

  ans = input("Continue? [Y/n]")
  if ans and ans.lower().startswith('n'):
    break







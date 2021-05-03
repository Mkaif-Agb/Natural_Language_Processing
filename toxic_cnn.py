import os 
import sys 
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Conv1D, MaxPooling1D, Embedding, Dense, Input, GlobalMaxPooling1D
from keras.models import Model
from keras.optimizers import Adam
from sklearn.metrics import roc_auc_score
import seaborn as sns
from tensorflow.python.keras.layers.recurrent import LSTM

# https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge
# Download the word vectors:
# http://nlp.stanford.edu/data/glove.6B.zip

MAX_SEQUENCE_LENGTH = 100
MAX_VOCAB_SIZE = 20000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2
BATCH_SIZE = 128
EPOCHS = 10

print(os.getcwd())
print("Loading Word Vectors")
word2vec = {}
with open(os.path.join('nlp/glove.6B/glove.6B.{}d.txt').format(EMBEDDING_DIM), encoding='utf8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        vec = np.array(values[1:])
        word2vec[word] = vec
print("Found {} of Word Vectors".format(len(word2vec)))


print("Loading Dataset")
train = pd.read_csv('nlp/toxic_train.csv')
# print(train.head())
# sns.heatmap(train.isnull(), cmap='coolwarm')
# plt.show()
sentences = train['comment_text'].fillna("DUMMY_VALUE").values
possible_labels = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
targets = train[possible_labels].values

# print(sentences[0], targets[0])

tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE)
tokenizer.fit_on_texts(sentences)
sequences = tokenizer.texts_to_sequences(sentences)

print("Max Sequence Length: {}".format(max(len(s) for s in sequences)))
print("Min Sequence Length: {}".format(min(len(s) for s in sequences)))
s = sorted(len(s) for s in sequences)
print("Median Sequence Length: {}".format(s[len(s) // 2]))
print("Max Word Index: {}".format(max(max(seq) for seq in sequences if len(seq) > 0 )))


# Word-Integer Mapping
word2idx = tokenizer.word_index
print("Found {} unique tokens".format(len(word2idx)))

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
print("Data Shape:", data.shape)


print("Filling Pre-Trained Embeddings")
num_words = min(MAX_VOCAB_SIZE, len(word2idx) + 1)
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
for word, i in word2idx.items():
    if i < MAX_VOCAB_SIZE:
        embedding_vector = word2vec.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector


embedding_layer = Embedding(num_words, EMBEDDING_DIM, weights=[embedding_matrix], 
                            input_length=MAX_SEQUENCE_LENGTH, trainable=False)

print('Building model...')

# train a 1D convnet with global maxpooling
input_ = Input(shape=(MAX_SEQUENCE_LENGTH,))
x = embedding_layer(input_)
x = Conv1D(128, 3, activation='relu')(x)
x = MaxPooling1D(3)(x)
x = Conv1D(128, 3, activation='relu')(x)
x = MaxPooling1D(3)(x)
x = Conv1D(128, 3, activation='relu')(x)
# x = LSTM(128, return_sequences=True)(x)
# x = LSTM(128, return_sequences=True)(x)
x = GlobalMaxPooling1D()(x)
x = Dense(128, activation='relu')(x)
output = Dense(len(possible_labels), activation='sigmoid')(x)

model = Model(input_, output)
model.compile(
  loss='binary_crossentropy',
  optimizer='rmsprop',
  metrics=['accuracy']
)

print('Training model...')
r = model.fit(
  data,
  targets,
  batch_size=BATCH_SIZE,
  epochs=EPOCHS,
  validation_split=VALIDATION_SPLIT
)

# 998/998 [==============================] - 1734s 2s/step - loss: 0.0491 - accuracy: 0.9921 - val_loss: 0.0490 - val_accuracy: 0.9938

plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['val_loss'], label='val_loss')
plt.legend()
plt.show()

# accuracies
plt.plot(r.history['accuracy'], label='acc')
plt.plot(r.history['val_accuracy'], label='val_acc')
plt.legend()
plt.show()

# plot the mean AUC over each label
p = model.predict(data)
aucs = []
for j in range(6):
    auc = roc_auc_score(targets[:,j], p[:,j])
    aucs.append(auc)
print(np.mean(aucs))
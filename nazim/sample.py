from __future__ import absolute_import, division, print_function

import os
import sys

import pickle
import tensorflow as tf
from six.moves import urllib

import tflearn
from tflearn.data_utils import *

import warnings
warnings.filterwarnings("ignore")

tf.logging.set_verbosity(tf.logging.ERROR)

path = "nazim_input.txt"
char_idx_file = 'char_idx.pickle'

maxlen = 25

char_idx = None
if os.path.isfile(char_idx_file):
  print('Loading previous char_idx')
  char_idx = pickle.load(open(char_idx_file, 'rb'))

#X, Y, char_idx = \
#    textfile_to_semi_redundant_sequences(path, seq_maxlen=maxlen, redun_step=3)

pickle.dump(char_idx, open(char_idx_file,'wb'))

g = tflearn.input_data([None, maxlen, len(char_idx)])
g = tflearn.lstm(g, 512, return_seq=True)
g = tflearn.dropout(g, 0.5)
g = tflearn.lstm(g, 512, return_seq=True)
g = tflearn.dropout(g, 0.5)
g = tflearn.lstm(g, 512)
g = tflearn.dropout(g, 0.5)
g = tflearn.fully_connected(g, len(char_idx), activation='softmax')
g = tflearn.regression(g, optimizer='adam', loss='categorical_crossentropy',
                       learning_rate=0.001)

m = tflearn.SequenceGenerator(g, dictionary=char_idx,
                              seq_maxlen=maxlen,
                              clip_gradients=5.0,
                              checkpoint_path='model_nazim')

seed = random_sequence_from_textfile(path, maxlen)
path_of_model = './nazim.final'

if len(sys.argv) > 1:
    path_of_model = sys.argv[1]

print(path_of_model)

if len(sys.argv) > 2:
    seed = sys.argv[2]

print(seed)

m.load(path_of_model)
print("print some samples")
print(m.generate(600, temperature=0.5, seq_seed=seed))

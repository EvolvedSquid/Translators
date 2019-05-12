""" English to French using Seq2Seq """

from __future__ import print_function

from keras.models import Model, load_model
from keras.layers import Input, LSTM, Dense
from keras.callbacks import LambdaCallback
import numpy as np
import random, os

def decode_sequence(input_seq):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0, target_token_index['\t']] = 1.

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_word = reverse_target_word_index[sampled_token_index]
        decoded_sentence += sampled_word + ' '

        # Exit condition: either hit max length
        # or find stop character.
        if (sampled_word == '\n' or
           len(decoded_sentence) > max_decoder_seq_length):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.

        # Update states
        states_value = [h, c]

    return decoded_sentence

def on_epoch_end(epoch, _):
    if epoch % 5 == 0:
        seq_index = random.randint(0, len(input_texts)-1)
        input_seq = encoder_input_data[seq_index: seq_index + 1]
        decoded_sentence = decode_sequence(input_seq)
        print('-')
        print('Input sentence:', input_texts[seq_index])
        print('Decoded sentence:', decoded_sentence)
    if epoch % 10 == 0 and save_model:
        model.save('Saves/eng2fra-%sx%s/model_%s.hdf5' % (num_samples, latent_dim, epoch))

def clean(text):
    return text.lower().replace('"', ' " ').replace('”', ' ” ').replace('“', ' “ ').replace(',', ' , ').replace('.', ' . ').replace('?', ' ? ').replace('!', ' ! ').replace('--', ' -- ').replace('- ', ' - ').replace(';', ' ; ').replace(':', ' : ').replace('=', ' = ').replace('(', ' ( ').replace(')', ' ) ').replace('[', ' [ ').replace(']', ' ] ').replace('{', ' { ').replace('}', ' } ')

def load(filepath):
    model = load_model(filepath)
    # encoder = load_model(load_from_encoder)
    # decoder = load_model(load_from_decoder)

    encoder_input = model.input[0]   # input_1
    encoder_output, state_h, state_c = model.layers[2].output   # lstm_1
    encoder_states = [state_h, state_c]

    encoder = Model(encoder_input, encoder_states, name='encoder')

    decoder_input = model.input[1]   # input_2
    decoder_state_input_h = Input(shape=(latent_dim,), name='decoder_inputH')
    decoder_state_input_c = Input(shape=(latent_dim,), name='decoder_inputC')
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_lstm = model.layers[3]
    decoder_outputs, state_h, state_c = decoder_lstm(
        decoder_input, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_dense = model.layers[4]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder = Model(
        [decoder_input] + decoder_states_inputs,
        [decoder_outputs] + decoder_states, name='decoder')
    
    return model, encoder, decoder

batch_size = None  # Batch size for training.
epochs = 200  # Number of epochs to train for.
latent_dim = 256  # Latent dimensionality of the encoding space.
num_samples = 2000  # Number of samples to train on.
save_model = False

# Path to the data txt file on disk.
data_path = 'Datasets/fra-eng/fra.txt'

# Vectorize the data.
input_texts = []
target_texts = []
english_vocab = set()
french_vocab = set()
with open(data_path, 'r', encoding='utf-8') as f:
    lines = f.read().split('\n')
for line in lines[: min(num_samples, len(lines) - 1)]:
    input_text, target_text = line.split('\t')
    input_text, target_text = clean(input_text).split(), clean(target_text).split()
    # We use "tab" as the "start sequence" character
    # for the targets, and "\n" as "end sequence" character.
    target_text = ['\t'] + target_text + ['\n']
    input_texts.append(input_text)
    target_texts.append(target_text)
    for word in input_text:
        if word not in english_vocab:
            english_vocab.add(word)
    for word in target_text:
        if word not in french_vocab:
            french_vocab.add(word)

# print(english_vocab); print(french_vocab)

english_vocab = sorted(list(english_vocab))
french_vocab = sorted(list(french_vocab))
num_encoder_tokens = len(english_vocab)
num_decoder_tokens = len(french_vocab)
max_encoder_seq_length = 10 #max([len(txt) for txt in input_texts])
max_decoder_seq_length = 12 #max([len(txt) for txt in target_texts])

print('Number of samples:', len(input_texts))
print('Number of unique input tokens:', num_encoder_tokens)
print('Number of unique output tokens:', num_decoder_tokens)
print('Max sequence length for inputs:', max_encoder_seq_length)
print('Max sequence length for outputs:', max_decoder_seq_length)

input_token_index = dict([(word, i) for i, word in enumerate(english_vocab)])
target_token_index = dict([(word, i) for i, word in enumerate(french_vocab)])

encoder_input_data = np.zeros((len(input_texts), max_encoder_seq_length, num_encoder_tokens), dtype='float32')
decoder_input_data = np.zeros((len(input_texts), max_decoder_seq_length, num_decoder_tokens), dtype='float32')
decoder_target_data = np.zeros((len(input_texts), max_decoder_seq_length, num_decoder_tokens), dtype='float32')

for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
    for t, word in enumerate(input_text):
        if t < max_encoder_seq_length:
            encoder_input_data[i, t, input_token_index[word]] = 1.
    for t, word in enumerate(target_text):
        if t < max_decoder_seq_length:
            # decoder_target_data is ahead of decoder_input_data by one timestep
            decoder_input_data[i, t, target_token_index[word]] = 1.
            if t > 0:
                # decoder_target_data will be ahead by one timestep
                # and will not include the start character.
                decoder_target_data[i, t - 1, target_token_index[word]] = 1.

# Define an input sequence and process it.
encoder_inputs = Input(shape=(None, num_encoder_tokens))
encoder = LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_inputs)
# We discard `encoder_outputs` and only keep the states.
encoder_states = [state_h, state_c]

# Set up the decoder, using `encoder_states` as initial state.
decoder_inputs = Input(shape=(None, num_decoder_tokens))
# We set up our decoder to return full output sequences,
# and to return internal states as well. We don't use the
# return states in the training model, but we will use them in inference.
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                     initial_state=encoder_states)
decoder_dense = Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# Define the model that will turn
# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

# Define sampling models
encoder_model = Model(encoder_inputs, encoder_states)

decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs] + decoder_states)

# model, encoder_model, decoder_model = load('Saves/eng2fra-2000x256/model_120.hdf5')

model.summary()

# Reverse-lookup token index to decode sequences back to
# something readable.
reverse_input_word_index = dict((i, char) for char, i in input_token_index.items())
reverse_target_word_index = dict((i, char) for char, i in target_token_index.items())

callback = LambdaCallback(on_epoch_end=on_epoch_end)

# Create save folder
if save_model:
    newpath = 'Saves/eng2fra-%sx%s' % (num_samples, latent_dim)
    if not os.path.exists(newpath):
        os.makedirs(newpath)

# Run training
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
          callbacks=[callback],
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.2)
# Save model
if save_model:
    model.save('Saves/eng2fra-%sx%s/model_%s.hdf5' % (num_samples, latent_dim, epochs))

# Next: inference mode (sampling).
# Here's the drill:
# 1) encode input and retrieve initial decoder state
# 2) run one step of decoder with this initial state
# and a "start of sequence" token as target.
# Output will be the next target token
# 3) Repeat with the current target token and current states


for _ in range(400):
    # Take one sequence (part of the training set)
    # for trying out decoding.
    seq_index = random.randint(0, len(input_texts)-1)
    input_seq = encoder_input_data[seq_index: seq_index + 1]
    decoded_sentence = decode_sequence(input_seq)
    print('-')
    print('Input sentence:', input_texts[seq_index])
    print('Decoded sentence:', decoded_sentence)

while True:
    sentence = clean(input('Input sentence: ')).split()
    temp_data = np.zeros((1, max_encoder_seq_length, num_encoder_tokens))
    for t, word in enumerate(sentence):
        if t < max_encoder_seq_length:
            temp_data[0, t, input_token_index[word]] = 1.
    print('Decoded sentence: ', decode_sequence(temp_data))
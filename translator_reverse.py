from __future__ import print_function

from keras.models import Model, load_model
from keras.layers import Input, LSTM, Dense
from keras.callbacks import LambdaCallback
import numpy as np
import random, os

def decode_sequence(input_seq):
    # For French
    states_value = encoder_model.predict(input_seq)

    target_seq = np.zeros((1, 1, num_french_tokens))
    target_seq[0, 0, french_token_index['\t']] = 1.

    stop_condition = False
    french_decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = f_decoder_model.predict(
            [target_seq] + states_value)

        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_word = french_vocab[sampled_token_index]
        french_decoded_sentence += sampled_word + ' '

        if (sampled_word == '\n' or
           len(french_decoded_sentence) > max_decoder_seq_length):
            stop_condition = True

        target_seq = np.zeros((1, 1, num_french_tokens))
        target_seq[0, 0, sampled_token_index] = 1.

        states_value = [h, c]

    # For German
    states_value = encoder_model.predict(input_seq)

    target_seq = np.zeros((1, 1, num_german_tokens))
    target_seq[0, 0, german_token_index['\t']] = 1.

    stop_condition = False
    german_decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = g_decoder_model.predict([target_seq] + states_value)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_word = german_vocab[sampled_token_index]
        german_decoded_sentence += sampled_word + ' '

        # Exit condition: either hit max length
        # or find stop character.
        if (sampled_word == '\n' or
        len(german_decoded_sentence) > max_decoder_seq_length):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, num_german_tokens))
        target_seq[0, 0, sampled_token_index] = 1.

        # Update states
        states_value = [h, c]
    
    return french_decoded_sentence, german_decoded_sentence

def on_epoch_end(epoch, _):
    if epoch % 5 == 0:
        seq_index = random.randint(0, len(ENG2fra)-1)
        input_seq = english2fra_input_data[seq_index: seq_index + 1]
        decoded_sentence = decode_sequence(input_seq)
        print('-')
        print('Input sentence:', ENG2fra[seq_index])
        print('Decoded sentence:', decoded_sentence)

def clean(text):
    return text.lower().replace('"', ' " ').replace('”', ' ” ').replace('“', ' “ ').replace(',', ' , ').replace('.', ' . ').replace('?', ' ? ').replace('!', ' ! ').replace('--', ' -- ').replace('- ', ' - ').replace(';', ' ; ').replace(':', ' : ').replace('=', ' = ').replace('(', ' ( ').replace(')', ' ) ').replace('[', ' [ ').replace(']', ' ] ').replace('{', ' { ').replace('}', ' } ')

def load(filepath):
    model = load_model(filepath)

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

def recover(eng2fra_file, eng2deu_file):
    # Load French file
    model = load_model(eng2fra_file)

    # Encoder for both
    encoder_input = model.input[0]   # input_1
    encoder_output, state_h, state_c = model.layers[2].output   # lstm_1
    encoder_states = [state_h, state_c]

    encoder = Model(encoder_input, encoder_states, name='encoder')

    # French decoder
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
    french_decoder = Model(
        [decoder_input] + decoder_states_inputs,
        [decoder_outputs] + decoder_states, name='decoder')
    
    # French model
    decoder_input = model.input[1]   # input_2
    decoder_state_input_h = Input(shape=(latent_dim,), name='decoder_inputH')
    decoder_state_input_c = Input(shape=(latent_dim,), name='decoder_inputC')
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_lstm = model.layers[3]
    decoder_outputs, state_h, state_c = decoder_lstm(
        decoder_input, initial_state=encoder_states)
    decoder_states = [state_h, state_c]
    decoder_dense = model.layers[4]
    decoder_outputs = decoder_dense(decoder_outputs)

    french_model = Model([encoder_input, decoder_input], decoder_outputs)

    # Load German file
    model = load_model(eng2deu_file)

    # German decoder
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
    german_decoder = Model(
        [decoder_input] + decoder_states_inputs,
        [decoder_outputs] + decoder_states, name='decoder')
    
    # German model
    decoder_input = model.input[1]   # input_2
    decoder_state_input_h = Input(shape=(latent_dim,), name='decoder_inputH')
    decoder_state_input_c = Input(shape=(latent_dim,), name='decoder_inputC')
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_lstm = model.layers[3]
    decoder_outputs, state_h, state_c = decoder_lstm(
        decoder_input, initial_state=encoder_states)
    decoder_states = [state_h, state_c]
    decoder_dense = model.layers[4]
    decoder_outputs = decoder_dense(decoder_outputs)

    german_model = Model([encoder_input, decoder_input], decoder_outputs)

    return french_model, german_model, encoder, french_decoder, german_decoder

batch_size = None  # Batch size for training.
epochs = 200  # Number of epochs to train for.
latent_dim = 256  # Latent dimensionality of the encoding space.
num_samples = 500  # Number of samples to train on (goes for each language).  Doubled samples because of reversed input.
save_model = True
load_from = None#('Saves/eng2fra&deu-750x256/eng2fra_200', 'Saves/eng2fra&deu-750x256/eng2deu_200')

# Path to the data txt file on disk.
data_path_1 = 'Datasets/fra-eng/fra.txt'
data_path_2 = 'Datasets/deu-eng/deu.txt'

# Vectorize the data.
ENG2fra = [] # English training to be translated to French
eng2FRA = [] # French target data
ENG2deu = [] # English training to be tanslated to German
eng2DEU = [] # German target data

english_vocab = set()
french_vocab = set()
german_vocab = set()

# Extract French data
with open(data_path_1, 'r', encoding='utf-8') as f:
    lines = f.read().split('\n')
for line in lines[: min(num_samples, len(lines) - 1)]:
    input_text, target_text = line.split('\t')
    input_text, target_text = clean(input_text).split(), clean(target_text).split()
    target_text = ['\t'] + target_text + ['\n']
    ENG2fra.append(input_text)
    ENG2fra.append(input_text[::-1]) # Reversed Sequence
    eng2FRA.append(target_text)
    eng2FRA.append(target_text) # Un-reversed target
    for word in input_text:
        if word not in english_vocab:
            english_vocab.add(word)
    for word in target_text:
        if word not in french_vocab:
            french_vocab.add(word)

# Extract German data
with open(data_path_2, 'r', encoding='utf-8') as f:
    lines = f.read().split('\n')
for line in lines[: min(num_samples, len(lines) - 1)]:
    input_text, target_text = line.split('\t')
    input_text, target_text = clean(input_text).split(), clean(target_text).split()
    target_text = ['\t'] + target_text + ['\n']
    ENG2deu.append(input_text)
    ENG2deu.append(input_text[::-1]) # Reversed Sequence
    eng2DEU.append(target_text)
    eng2DEU.append(target_text) # Un-reversed target
    for word in input_text:
        if word not in english_vocab:
            english_vocab.add(word)
    for word in target_text:
        if word not in german_vocab:
            german_vocab.add(word)

print(english_vocab); print(french_vocab); print(german_vocab)

english_vocab = sorted(list(english_vocab))
french_vocab = sorted(list(french_vocab))
german_vocab = sorted(list(german_vocab))

num_english_tokens = len(english_vocab)
num_french_tokens = len(french_vocab)
num_german_tokens = len(german_vocab)

max_encoder_seq_length = max([len(txt) for txt in ENG2fra] + [len(txt) for txt in ENG2deu])
max_decoder_seq_length = max([len(txt) for txt in eng2FRA] + [len(txt) for txt in eng2DEU])

print('Number of samples:', len(ENG2fra), len(ENG2deu))
print('Number of unique input tokens:', num_english_tokens)
print('Number of unique output tokens:', num_french_tokens, num_german_tokens)
print('Max sequence length for inputs:', max_encoder_seq_length)
print('Max sequence length for outputs:', max_decoder_seq_length)

english_token_index = dict([(word, i) for i, word in enumerate(english_vocab)])
french_token_index = dict([(word, i) for i, word in enumerate(french_vocab)])
german_token_index = dict([(word, i) for i, word in enumerate(german_vocab)])

print(len(ENG2fra), len(eng2FRA), len(ENG2deu), len(eng2DEU))


# Encode French data
english2fra_input_data = np.zeros((num_samples*2, max_encoder_seq_length, num_english_tokens), dtype='float32')
french_input_data = np.zeros((num_samples*2, max_decoder_seq_length, num_french_tokens), dtype='float32')
french_target_data = np.zeros((num_samples*2, max_decoder_seq_length, num_french_tokens), dtype='float32')

for i, (input_text, target_text) in enumerate(zip(ENG2fra, eng2FRA)):
    for t, word in enumerate(input_text):
        if t < max_encoder_seq_length:
            english2fra_input_data[i, t, english_token_index[word]] = 1.
    for t, word in enumerate(target_text):
        if t < max_decoder_seq_length:
            # decoder_target_data is ahead of decoder_input_data by one timestep
            french_input_data[i, t, french_token_index[word]] = 1.
            if t > 0:
                # decoder_target_data will be ahead by one timestep
                # and will not include the start character.
                french_target_data[i, t - 1, french_token_index[word]] = 1.

# Encode German data
english2deu_input_data = np.zeros((num_samples*2, max_encoder_seq_length, num_english_tokens), dtype='float32')
german_input_data = np.zeros((num_samples*2, max_decoder_seq_length, num_german_tokens), dtype='float32')
german_target_data = np.zeros((num_samples*2, max_decoder_seq_length, num_german_tokens), dtype='float32')

for i, (input_text, target_text) in enumerate(zip(ENG2deu, eng2DEU)):
    for t, word in enumerate(input_text):
        if t < max_encoder_seq_length:
            english2deu_input_data[i, t, english_token_index[word]] = 1.
    for t, word in enumerate(target_text):
        if t < max_decoder_seq_length:
            # decoder_target_data is ahead of decoder_input_data by one timestep
            german_input_data[i, t, german_token_index[word]] = 1.
            if t > 0:
                # decoder_target_data will be ahead by one timestep
                # and will not include the start character.
                german_target_data[i, t - 1, german_token_index[word]] = 1.

if load_from == None:
    # Define an input sequence and process it.
    encoder_inputs = Input(shape=(None, num_english_tokens))
    encoder = LSTM(latent_dim, return_state=True)
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    # We discard `encoder_outputs` and only keep the states.
    encoder_states = [state_h, state_c]

    # French decoder
    # Set up the decoder, using `encoder_states` as initial state.
    f_decoder_inputs = Input(shape=(None, num_french_tokens), name='french_encoder_input')
    # We set up our decoder to return full output sequences,
    # and to return internal states as well. We don't use the
    # return states in the training model, but we will use them in inference.
    f_decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
    f_decoder_outputs, _, _ = f_decoder_lstm(f_decoder_inputs,
                                        initial_state=encoder_states)
    f_decoder_dense = Dense(num_french_tokens, activation='softmax')
    f_decoder_outputs = f_decoder_dense(f_decoder_outputs)

    # German decoder
    # Set up the decoder, using `encoder_states` as initial state.
    g_decoder_inputs = Input(shape=(None, num_german_tokens))
    # We set up our decoder to return full output sequences,
    # and to return internal states as well. We don't use the
    # return states in the training model, but we will use them in inference.
    g_decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
    g_decoder_outputs, _, _ = g_decoder_lstm(g_decoder_inputs,
                                        initial_state=encoder_states)
    g_decoder_dense = Dense(num_german_tokens, activation='softmax')
    g_decoder_outputs = g_decoder_dense(g_decoder_outputs)

    # Define the model that will turn
    # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
    eng2fra_model = Model([encoder_inputs, f_decoder_inputs], f_decoder_outputs)
    eng2deu_model = Model([encoder_inputs, g_decoder_inputs], g_decoder_outputs)

    # Define sampling models
    encoder_model = Model(encoder_inputs, encoder_states)

    # French decoder as model
    decoder_input = eng2fra_model.input[1]   # input_2
    decoder_state_input_h = Input(shape=(latent_dim,), name='decoder_inputH')
    decoder_state_input_c = Input(shape=(latent_dim,), name='decoder_inputC')
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_lstm = eng2fra_model.layers[3]
    decoder_outputs, state_h, state_c = decoder_lstm(
        decoder_input, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_dense = eng2fra_model.layers[4]
    decoder_outputs = decoder_dense(decoder_outputs)
    f_decoder_model = Model(
        [decoder_input] + decoder_states_inputs,
        [decoder_outputs] + decoder_states, name='decoder')

    # German decoder as model
    decoder_input = eng2deu_model.input[1]   # input_2
    decoder_state_input_h = Input(shape=(latent_dim,), name='decoder_inputH')
    decoder_state_input_c = Input(shape=(latent_dim,), name='decoder_inputC')
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_lstm = eng2deu_model.layers[3]
    decoder_outputs, state_h, state_c = decoder_lstm(
        decoder_input, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_dense = eng2deu_model.layers[4]
    decoder_outputs = decoder_dense(decoder_outputs)
    g_decoder_model = Model(
        [decoder_input] + decoder_states_inputs,
        [decoder_outputs] + decoder_states, name='decoder')
else:
    # Saves/eng2fra&deu-750x256/eng2fra_XX, Saves/eng2fra&deu-750x256/eng2deu_XX
    eng2fra_model, eng2deu_model, encoder_model, f_decoder_model, g_decoder_model = recover(load_from[0], load_from[1])

eng2fra_model.summary()
eng2deu_model.summary()
print(max_encoder_seq_length, max_decoder_seq_length)

# Reverse-lookup token index to decode sequences back to something readable
reverse_input_word_index = dict((i, char) for char, i in english_token_index.items())
reverse_target_word_index = dict((i, char) for char, i in french_token_index.items())

# Create save folder
if save_model:
    newpath = 'Saves/eng2fra&deu_reversed-%sx%s' % (num_samples, latent_dim)
    if not os.path.exists(newpath):
        os.makedirs(newpath)

# Run training
eng2fra_model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
eng2deu_model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

callback = LambdaCallback(on_epoch_end=on_epoch_end)

# eng2fra_model.fit([english2fra_input_data, french_input_data], french_target_data, # WORKS!
#           callbacks=[callback],
#           batch_size=batch_size,
#           epochs=epochs,
#           validation_split=0.2)

for epoch in range(epochs):
    print('Epoch %s/%s' % (epoch+1, epochs))
    eng2fra_model.fit([english2fra_input_data, french_input_data], french_target_data,
          batch_size=batch_size,
          epochs=1,
          verbose=0)
    eng2deu_model.fit([english2deu_input_data, german_input_data], german_target_data,
          batch_size=batch_size,
          epochs=1,
          verbose=0)

    if epoch % 5 == 0:
        seq_index = random.randint(0, len(ENG2fra)-1)
        input_seq = english2fra_input_data[seq_index: seq_index + 1]
        decoded_sentence = decode_sequence(input_seq)
        print('-')
        print('Input sentence:', ENG2fra[seq_index])
        print('Decoded sentence:', decoded_sentence)
        print('French - Loss: %s, Acc: %s ' % tuple(eng2fra_model.test_on_batch([english2fra_input_data, french_input_data], french_target_data)))
        print('German - Loss: %s, Acc: %s ' % tuple(eng2deu_model.test_on_batch([english2deu_input_data, german_input_data], german_target_data)))
        print('-')


    if epoch % 10 == 0 and save_model:
        eng2fra_model.save('Saves/eng2fra&deu_reversed-%sx%s/eng2fra_%s' % (num_samples, latent_dim, epoch))
        eng2deu_model.save('Saves/eng2fra&deu_reversed-%sx%s/eng2deu_%s' % (num_samples, latent_dim, epoch))

# Save model
if save_model:
    eng2fra_model.save('Saves/eng2fra&deu_reversed-%sx%s/eng2fra_%s' % (num_samples, latent_dim, epochs))
    eng2deu_model.save('Saves/eng2fra&deu_reversed-%sx%s/eng2deu_%s' % (num_samples, latent_dim, epochs))

# eng2fra_model, eng2deu_model, encoder, f_decoder_model, g_decoder_model = recover('Saves/eng2fra&deu-500x256/eng2fra_12', 'Saves/eng2fra&deu-500x256/eng2deu_12')

for _ in range(75):
    # Take one sequence (part of the training set)
    # for trying out decoding.
    seq_index = random.randint(0, len(ENG2fra)-1)
    input_seq = english2fra_input_data[seq_index: seq_index + 1]
    decoded_sentence = decode_sequence(input_seq)
    print('-')
    print('Input sentence:', ENG2fra[seq_index])
    print('Decoded sentence:', decoded_sentence)

while True:
    sentence = clean(input('Input sentence: ')).split()
    temp_data = np.zeros((1, max_encoder_seq_length, num_english_tokens))
    for t, word in enumerate(sentence):
        if t < max_encoder_seq_length:
            try:
                temp_data[0, t, english_token_index[word]] = 1.
            except:
                print("Unrecognized Word: '%s'" % word)
                break
    print('Decoded sentence: ', decode_sequence(temp_data))

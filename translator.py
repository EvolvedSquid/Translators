import re, io, collections, random
from keras.callbacks import LambdaCallback, ModelCheckpoint
from keras.models import Sequential, load_model, Model
from keras.layers import Dense, Flatten, Reshape, Input
from keras.layers import LSTM, GRU, Bidirectional, RepeatVector, TimeDistributed
from keras.optimizers import RMSprop, Adam
from keras.utils.data_utils import get_file
from keras.utils import to_categorical
from keras import backend as K
import numpy as np
import tensorflow as tf

def clean(text):
    return text.lower().replace('"', ' " ').replace('”', ' ” ').replace('“', ' “ ').replace(',', ' , ').replace('.', ' . ').replace('?', ' ? ').replace('!', ' ! ').replace('--', ' -- ').replace('- ', ' - ').replace(';', ' ; ').replace(':', ' : ').replace('=', ' = ').replace('(', ' ( ').replace(')', ' ) ').replace('[', ' [ ').replace(']', ' ] ').replace('{', ' { ').replace('}', ' } ')

def write():
    num = random.randint(0, len(coupled_eng2fra[0])-1)
    sent_x = english_data[num]
    sent_y_t1 = french_target_data[num]
    sentence = coupled_eng2fra[0][num]
    translated_sentence = coupled_eng2fra[1][num]
    sent_x = sent_x.reshape((1, maxlen, eng_tokens))
    states = encoder.predict(sent_x)
    sequence = np.zeros((1, 1, fra_tokens))
    sequence[0, 0, french_vocab['<START>']] = 1

    is_writing = True

    decoded_sentence = []
    while is_writing:
        output_tokens, h, c = decoder.predict([sequence] + states)
        sampled_word_index = np.argmax(output_tokens[0, -1, :])
        sampled_word = reverse_fra_vocab[sampled_word_index]
        decoded_sentence.append(sampled_word)
        if sampled_word == '<STOP>' or len(decoded_sentence) > maxlen:
            is_writing = False

        target_seq = np.zeros((1, 1, fra_tokens))
        target_seq[0, 0, sampled_word_index] = 1.

        states = [h, c]
    print('Encoding: %s - %s.  Ground truth: %s' % (sentence, decoded_sentence, translated_sentence))  

def callback(epoch, _):
    if epoch % 5 == 0:
        write()

filepath1 = 'Datasets/fra-eng/fra.txt'
filepath2 = 'Datasets/deu-eng/deu.txt'


with open(filepath1, 'r', encoding='utf-8') as f:
    lines = f.read().split('\n')

coupled_eng2fra = [
    [], # English sentences
    []  # French translations
]

english_vocab = {'<UNK>': 0}
french_vocab = {}

epochs = 1500
maxlen = 8
latent_dim = 650
difficulty = 1000 # Number of examples shown 

for line in lines[:difficulty]:
    input_text, target_text = line.split('\t')
    input_text, target_text = clean(input_text).split()[:maxlen], clean(target_text).split()
    target_text = ['<START>'] + target_text[:maxlen-2] + ['<STOP>']

    print(input_text, target_text)

    coupled_eng2fra[0].append(input_text)
    coupled_eng2fra[1].append(target_text)

    for eng_word in input_text:
        if eng_word not in english_vocab.keys():
            english_vocab[eng_word] = len(english_vocab)
    for fra_word in target_text:
        if fra_word not in french_vocab.keys():
            french_vocab[fra_word] = len(french_vocab)

print(english_vocab)
print(french_vocab)

eng_tokens = len(english_vocab)
fra_tokens = len(french_vocab)

reverse_eng_vocab = {v: k for k, v in english_vocab.items()}
reverse_fra_vocab = {v: k for k, v in french_vocab.items()}

english_data = np.zeros((len(coupled_eng2fra[0]), maxlen, eng_tokens))
french_data = np.zeros((len(coupled_eng2fra[1]), maxlen, fra_tokens))
french_target_data = np.zeros((len(coupled_eng2fra[1]), maxlen, fra_tokens))

for i, (eng_text, fra_text) in enumerate(zip(coupled_eng2fra[0], coupled_eng2fra[1])):
    for t, word in enumerate(eng_text):
        if word in english_vocab.keys():
            english_data[i, t, english_vocab[word]] = 1
        else:
            english_data[i, t, '<UNK>'] = 1
    for t, word in enumerate(fra_text):
        french_data[i, t, french_vocab[word]] = 1

        if t > 1:
            french_target_data[i, t-1, french_vocab[word]] = 1


encoder_input = Input(shape=(None, eng_tokens), name='encoder_input') # Encoder input, x data
bottleneck = LSTM(latent_dim, return_state=True, name='bottleneck') # Bottleneck (add more layers before this???)
encoder_outputs, state_h, state_c = bottleneck(encoder_input) # Calls bottleneck on input to get outputs

encoder_states = [state_h, state_c]

decoder_input = Input(shape=(None, fra_tokens), name='decoder_input') # Decoder input, output data shifted by timestep
decoder_lstm0 = LSTM(latent_dim, return_sequences=True, return_state=True, name='decoder_lstm0') # Decoder lstm (add more layers after this???)
decoder_outputs, _, _ = decoder_lstm0(decoder_input, initial_state=encoder_states) # Calls decoder lstm on decoder inputs, with encoder output states as init state
decoder_dense0 = Dense(fra_tokens, activation='softmax', name='decoder_dense0') # Turns output given by decoder lstm to a probability map of next word
decoder_outputs = decoder_dense0(decoder_outputs) # Calls dense layer on current outputs for new outputs

#               sentence_x      sentence_y_t1      sentence_y             
model = Model([encoder_input, decoder_input], decoder_outputs)

encoder = Model(encoder_input, encoder_states)

decoder_state_input_h = Input(shape=(latent_dim,), name='decoder_inputH') # Creates inputs for initial states
decoder_state_input_c = Input(shape=(latent_dim,), name='decoder_inputC')
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_outputs, state_h, state_c = decoder_lstm0(decoder_input, initial_state=decoder_states_inputs) # Repteats code from above (would have to add new layers for decoder here too
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense0(decoder_outputs)

decoder = Model([decoder_input] + decoder_states_inputs, [decoder_outputs] + decoder_states)


model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy', 'categorical_accuracy'])

on_epoch_end = LambdaCallback(on_epoch_end=callback)

model.fit([english_data, french_data], french_target_data,
          callbacks=[on_epoch_end],
          batch_size=None,
          epochs=epochs,
          validation_split=0.2)

for _ in range(50):
    write()

model.save('Saves/translator_rmsprop.hdf5', overwrite=False)
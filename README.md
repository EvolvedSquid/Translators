# Translators
Translates English to French and German using an LSTM.
The goal of the project is to create an LSTM an understanding of natural language by forcing two seperate decoders work with a single encoder.

`translator.py` is the base file for training an LSTM.  `translator_reverse.py` trains the LSTM with both regular and reversed input sequences.

See Datasets/DOWNLOAD.md for instructions to install datasets.
All credit to the [Tatoeba Project](tatoeba.org/home), liscenced under Creative Commons, for data.

A lot of code was borrowed from https://github.com/keras-team/keras/blob/master/examples/lstm_seq2seq.py by Francois Chollet.  The code was modified to be word level and to work with multiple languages at once as well as other changes.

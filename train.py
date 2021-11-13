""" This module prepares midi file data and feeds it to the neural
    network for training """
import glob
import pickle
import numpy
from music21 import converter, instrument, note, chord
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import SimpleRNN
from keras.layers import GRU
from keras.layers import Activation
from keras.layers import BatchNormalization as BatchNorm
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from keras.layers import LeakyReLU
import sys

def train_network(model_type):

    """ Train a Neural Network to generate music """
    notes = get_notes()

    # get amount of pitch names
    pitchnames = set()
    for midi_notes in notes:

        #extract all pitches
        for note in midi_notes:
            pitchnames.add(note)

    pitchnames = sorted(pitchnames)
    # Get all pitch names
    n_vocab = len(set(pitchnames))

    network_input, network_output = prepare_sequences(notes, n_vocab)

    model = None

    if model_type == "lstm":
        model = create_lstm(network_input, n_vocab)
    elif model_type == "rnn":
        model = create_rnn(network_input, n_vocab)
    elif model_type == "gru":
        model = create_gru(network_input, n_vocab)
    else:
        print("Invalid Model Type Input")
        return

    train(model, network_input, network_output, model_type)

def get_notes():
    """ Get all the notes and chords from the midi files in the ./midi_songs directory """
    notes = []

    for file in glob.glob("midi_songs/*.mid"):
        midi = converter.parse(file)

        print("Parsing %s" % file)

        midi_notes = []
        notes_to_parse = midi.flat.notes

        for element in notes_to_parse:
            if isinstance(element, note.Note):
                midi_notes.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                midi_notes.append('.'.join(str(n) for n in element.normalOrder))
        notes.append(midi_notes)

    with open('data/notes', 'wb') as filepath:
        pickle.dump(notes, filepath)

    return notes

def prepare_sequences(notes, n_vocab):
    """ Prepare the sequences used by the Neural Network """
    sequence_length = 100

    #extract all existing pitches from the notes
    pitchnames = set()
    for midi_notes in notes:

        #extract all pitches
        for note in midi_notes:
            pitchnames.add(note)

    pitchnames = sorted(pitchnames)

     # create a dictionary to map pitches to integers
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

    network_input = []
    network_output = []

    for midi_notes in notes:
        #create network input and output
        for i in range(0, len(midi_notes) - sequence_length, 1):
            sequence_in = midi_notes[i:i + sequence_length]
            sequence_out = midi_notes[i + sequence_length]
            network_input.append([note_to_int[char] for char in sequence_in])
            network_output.append(note_to_int[sequence_out])

    n_patterns = len(network_input)

    # reshape the input into a format compatible with LSTM layers
    network_input = numpy.reshape(network_input, (n_patterns, sequence_length, 1))
    # normalize input
    network_input = network_input / float(n_vocab)

    network_output = np_utils.to_categorical(network_output)

    return (network_input, network_output)

def create_lstm(network_input, n_vocab):
    """ create the structure of lstm """
    model = Sequential()
    model.add(LSTM(
        512,
        input_shape=(network_input.shape[1], network_input.shape[2]),
        recurrent_dropout=0.3,
        return_sequences=True
    ))
    model.add(LSTM(512, return_sequences=True, recurrent_dropout=0.3,))
    model.add(LSTM(512))
    model.add(BatchNorm())
    model.add(Dropout(0.3))
    model.add(Dense(256))
    model.add(LeakyReLU(alpha=0.3))
    model.add(BatchNorm())
    model.add(Dropout(0.3))
    model.add(Dense(n_vocab))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    return model

def create_rnn(network_input, n_vocab):
    """create the structure of simple rnn"""
    model = Sequential()
    model.add(SimpleRNN(
        512,
        input_shape=(network_input.shape[1], network_input.shape[2]),
        recurrent_dropout=0.3,
        return_sequences=True
    ))
    model.add(SimpleRNN(512, return_sequences=True, recurrent_dropout=0.3,))
    model.add(SimpleRNN(512))
    model.add(BatchNorm())
    model.add(Dropout(0.3))
    model.add(Dense(256))
    model.add(LeakyReLU(alpha=0.3))
    model.add(BatchNorm())
    model.add(Dropout(0.3))
    model.add(Dense(n_vocab))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    return model

def create_gru(network_input, n_vocab):
    """create the structure of simple rnn"""
    model = Sequential()
    model.add(GRU(
        512,
        input_shape=(network_input.shape[1], network_input.shape[2]),
        recurrent_dropout=0.3,
        return_sequences=True
    ))
    model.add(GRU(512, return_sequences=True, recurrent_dropout=0.3,))
    model.add(GRU(512))
    model.add(BatchNorm())
    model.add(Dropout(0.3))
    model.add(Dense(256))
    model.add(LeakyReLU(alpha=0.3))
    model.add(BatchNorm())
    model.add(Dropout(0.3))
    model.add(Dense(n_vocab))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    return model

def train(model, network_input, network_output, model_type):
    """ train the neural network """
    file_path = None
    if model_type == "lstm":
        filepath = "touhou_lstm.hdf5"
    elif model_type == "rnn":
        filepath = "touhou_rnn.hdf5"
    elif model_type == "gru":
        filepath = "touhou_gru.hdf5"

    checkpoint = ModelCheckpoint(
        filepath,
        monitor='loss',
        verbose=0,
        save_best_only=True,
        mode='min'
    )
    callbacks_list = [checkpoint]

    model.fit(network_input, network_output, epochs=200, batch_size=128, callbacks=callbacks_list)

if __name__ == '__main__':
    train_network(sys.argv[1])

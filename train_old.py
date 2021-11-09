import glob
import numpy as np
from music21 import converter, instrument, note, chord
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Activation
from keras.layers import BatchNormalization as BatchNorm
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint

def pre_process_notes():
    """Read in the prepared midi files via music21 and extract the notes and chords"""

    notes = []

    #Read in midi files and create a note sequence
    for file in glob.glob("midi_songs_og/*.mid"):
        print(file)
        midi_file = converter.parse(file)

        midi_notes = []
        notes_to_parse = midi_file.flat.notes

        for element in notes_to_parse:
            #the element is a note
            if isinstance(element, note.Note):
                midi_notes.append(str(element.pitch))
            #the element is a chord
            elif isinstance(element, chord.Chord):
                midi_notes.append('.'.join(str(n) for n in element.pitches))

        print(midi_notes)
        notes.append(midi_notes)

    return notes

def prepare_sequences(notes):
    """Prepare the input and output sequences for the network"""

    sequence_length = 100
    network_input = []
    network_output = []

    #extract all existing pitches from the notes
    pitchnames = set()
    for midi_notes in notes:

        #extract all pitches
        for note in midi_notes:
            pitchnames.add(note)

    pitchnames = sorted(pitchnames)
    n_vocab = len(pitchnames)

    # create a dictionary to map pitches to integers
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

    for midi_notes in notes:
        #create network input and output
        for i in range(0, len(midi_notes) - sequence_length, 1):
            sequence_in = midi_notes[i:i + sequence_length]
            sequence_out = midi_notes[i + sequence_length]
            network_input.append([note_to_int[char] for char in sequence_in])
            network_output.append(note_to_int[sequence_out])

    n_patterns = len(network_input)

    # reshape the input into a format compatible with the model
    network_input = np.reshape(network_input, (n_patterns, sequence_length, 1))
    # normalize input
    network_input = network_input / float(n_vocab)

    #convert the output to its binary matrix representation
    network_output = np_utils.to_categorical(network_output)

    return (network_input, network_output, n_vocab)

def create_three_layer_LSTM(network_input, n_vocab):
    """Create the structure of a three layer LSTM"""
    model = Sequential()
    model.add(LSTM(
        512,
        input_shape=(network_input.shape[1], network_input.shape[2]),
        recurrent_dropout=0.75,
        return_sequences=True
    ))
    model.add(LSTM(512, return_sequences=True, recurrent_dropout=0.75,))
    model.add(LSTM(512))
    model.add(BatchNorm())
    model.add(Dropout(0.75))
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(BatchNorm())
    model.add(Dropout(0.75))
    model.add(Dense(n_vocab))
    model.add(Activation('relu'))
    model.compile(loss='mse', optimizer='rmsprop')

    return model

def train(model, network_input, network_output):
    """train the neural network"""
    filepath = "trained-model.hdf5"
    checkpoint = ModelCheckpoint(
        filepath,
        monitor='loss',
        verbose=0,
        save_best_only=True,
        mode='min'
    )
    callbacks_list = [checkpoint]

    model.fit(network_input, network_output, epochs=64, batch_size=32, callbacks=callbacks_list)

def train_start():

    notes = pre_process_notes()
    network_input, network_output, n_vocab = prepare_sequences(notes)
    model = create_three_layer_LSTM(network_input, n_vocab)
    train(model, network_input, network_output)

if __name__ == "__main__":
    train_start()

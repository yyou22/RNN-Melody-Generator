""" This module generates notes for a midi file using the
    trained neural network """
import pickle
import numpy
from music21 import instrument, note, stream, chord
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import SimpleRNN
from keras.layers import GRU
from keras.layers import BatchNormalization as BatchNorm
from keras.layers import Activation
from keras.layers import LeakyReLU
import random
import sys

def generate(model_type):
    """ Generate a piano midi file """
    #load the notes used to train the model
    with open('data/notes', 'rb') as filepath:
        notes = pickle.load(filepath)

    #extract all existing pitches from the notes
    pitchnames = set()
    for midi_notes in notes:

        #extract all pitches
        for note in midi_notes:
            pitchnames.add(note)

    pitchnames = sorted(pitchnames)
    # Get all pitch names
    n_vocab = len(set(pitchnames))

    #load the input sequences
    with open('input/input_notes', 'rb') as filepath:
        input_notes = pickle.load(filepath)

    network_input, normalized_input = prepare_sequences(notes, input_notes, pitchnames, n_vocab)

    model = None

    if model_type == "lstm":
        model = create_lstm(normalized_input, n_vocab)
    elif model_type == "rnn":
        model = create_rnn(normalized_input, n_vocab)
    elif model_type == "gru":
        model = create_gru(normalized_input, n_vocab)
    else:
        print("Invalid Model Type Input")
        return

    prediction_output = generate_notes(model, network_input, pitchnames, n_vocab)
    create_midi(prediction_output, model_type)

def prepare_sequences(notes, input_notes, pitchnames, n_vocab):
    """ Prepare the sequences used by the Neural Network """
    # map between notes and integers and back
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

    #sequence_length = 32
    network_input = []
    for midi_notes in input_notes:
        sequence_length = len(midi_notes)
        #create network input and output
        for i in range(0, len(midi_notes) + 1 - sequence_length, 1):
            sequence_in = midi_notes[i:i + sequence_length]
            network_input.append([note_to_int[char] for char in sequence_in])

    n_patterns = len(network_input)
    # reshape the input into a format compatible with LSTM layers
    normalized_input = numpy.reshape(network_input, (n_patterns, sequence_length, 1))
    # normalize input
    normalized_input = normalized_input / float(n_vocab)

    return (network_input, normalized_input)

def create_lstm(network_input, n_vocab):
    """ create the structure of the neural network """
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

    # Load the weights to each node
    model.load_weights('touhou_lstm.hdf5')

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

    # Load the weights to each node
    model.load_weights('touhou_rnn.hdf5')

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

    # Load the weights to each node
    model.load_weights('touhou_gru.hdf5')

    return model

def generate_notes(model, network_input, pitchnames, n_vocab):
    """ Generate notes from the neural network based on a sequence of notes """
    # pick a random sequence from the input as a starting point for the prediction
    #start = numpy.random.randint(0, len(network_input)-1)
    start = 0
    print("starting sequence: " + str(start))

    int_to_note = dict((number, note) for number, note in enumerate(pitchnames))

    pattern = network_input[start]
    prediction_output = []

    # generate notes
    for note_index in range(100):
        prediction_input = numpy.reshape(pattern, (1, len(pattern), 1))
        prediction_input = prediction_input / float(n_vocab)

        prediction = model.predict(prediction_input, verbose=0)
        #print(prediction)

        index = numpy.argmax(prediction)
        print(index)
        result = int_to_note[index]
        prediction_output.append(result)

        pattern.append(index)
        pattern = pattern[1:len(pattern)]

    return prediction_output

def create_midi(prediction_output, model_type):
    """ convert the output from the prediction to notes and create a midi file
        from the notes """
    offset = 0
    output_notes = []

    # create note and chord objects based on the values generated by the model
    for pattern in prediction_output:
        # pattern is a chord
        #duration = random.choice([0.25, 0.5])
        duration = 0.25
        if ('.' in pattern) or pattern.isdigit():
            notes_in_chord = pattern.split('.')
            notes = []
            for current_note in notes_in_chord:
                new_note = note.Note(int(current_note))
                new_note.storedInstrument = instrument.Piano()
                new_note.quarterLength = duration
                notes.append(new_note)
            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            output_notes.append(new_chord)
        # pattern is a note
        else:
            new_note = note.Note(pattern)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            new_note.quarterLength = duration
            output_notes.append(new_note)

        # increase offset each iteration so that notes do not stack
        offset += duration

    midi_stream = stream.Stream(output_notes)

    if model_type == "lstm":
        midi_stream.write('midi', fp='output_lstm.mid')
    elif model_type == "rnn":
        midi_stream.write('midi', fp='output_rnn.mid')
    elif model_type == "gru":
        midi_stream.write('midi', fp='output_gru.mid')

if __name__ == '__main__':
    generate(sys.argv[1])

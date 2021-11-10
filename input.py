import glob
import pickle
import numpy
from music21 import converter, instrument, note, chord
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Activation
from keras.layers import BatchNormalization as BatchNorm
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from keras.layers import LeakyReLU

def process_input():
    """Process the sequences that will be used to generate melodies"""
    input_notes = []

    for file in glob.glob("input/*.mid"):
        midi = converter.parse(file)

        print("Parsing %s" % file)

        midi_notes = []
        notes_to_parse = midi.flat.notes

        for element in notes_to_parse:
            if isinstance(element, note.Note):
                midi_notes.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                midi_notes.append('.'.join(str(n) for n in element.normalOrder))
        input_notes.append(midi_notes)

    with open('input/input_notes', 'wb') as filepath:
        pickle.dump(input_notes, filepath)

    return input_notes

if __name__ == '__main__':
    process_input()

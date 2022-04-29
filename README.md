
# Melody AI: RNN-based Melody Generator

Author:
Yuzhe You

In this study, three RNN-based models are implemented with different RNN variants (vanilla RNN, LSTM & GRU). A set of MIDI files are collected and partitioned into single-instrument melodies to be used as a starting sequence for the models.

## Environment
- Python3
- Keras Tensorflow

## Getting started

### Data Processing & Model Training
* Place MIDI files (for training) inside the midi_songs folder
* Run `python train.py [rnn/gru/lstm]`
* Using a GRU-based or LSTM-based model is recommended for better results due to vanilla RNN suffering from the vanishing gradient problem

### Input Sequence Preprocessing
* Place the starting sequence (in the form of a MIDI file) in the input folder
* Run `python input.py`
* This will process all the sequences that will be used to generate melodies from the MIDI files located within the input folder

### Melody Generation
* Run `python predict.py [rnn/gru/lstm]`
* This will generate a melody of sequence length 100 using trained model

## Sample Outputs
A playlist of sample outputs can be heard here:
[Melody AI SoundCloud Playlist](https://soundcloud.com/yyou22/sets/melody-ai).

## References
The implementation of this project is partially based on the implementation from [How to Generate Music using a LSTM Neural Network in Keras](https://towardsdatascience.com/how-to-generate-music-using-a-lstm-neural-network-in-keras-68786834d4c5).
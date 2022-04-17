
# Creative AI: RNN-based Melody Generator

Author:
Yuzhe You

In this study, three RNN-based models are implemented with different RNN variants (vanilla RNN, LSTM & GRU). A set of MIDI files are collected and partitioned into single-instrument melodies to be used as a starting sequence for the models.

## Environment
- Python3
- Keras Tensorflow

## Getting started
### Data Preprocessing
* run `python input.py`
* This will process all the sequences that will be used to generate melodies from the MIDI files located within the input folder

### Model Training
* run `python train.py [rnn/gru/lstm]`
* Using a GRU-based or LSTM-based model is recommended for better results due to vanilla RNN suffering from the vanishing gradient problem

### Melody Generation
* run `python predict.py [rnn/gru/lstm]`
* Generate a melody of sequence length 100 using trained model

## References
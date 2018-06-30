# pix2pix

## Description
  * This is a Tensorflow implementaion of Audio source separation (mixture to vocal) using the [pix2pix](https://arxiv.org/abs/1611.07004). I pre-processed raw data(mixture and vocal pair dataset) to spectrogram that can be treated as 2-dimensional image, then train the model. See the file `hyperparams.py` for the detailed hyperparameters.

## Requirements
  * NumPy >= 1.11.1
  * TensorFlow >= 1.0.0
  * librosa

## Data
I used DSD100 dataset which consists of pairs of mixture audio files and vocal audio files. The complete dataset (~14 GB) can be downloaded [here](http://liutkus.net/DSD100.zip). 

## File description
  * `hyperparams.py` includes all hyper parameters that are needed.
  * `data.py` loads training data and preprocess it into units of raw data sequences.
  * `modules.py` contains all methods, building blocks and skip connections for networks.
  * `networks.py` builds networks.
  * `train.py` is for training.

## Training the network
  * STEP 1. Adjust hyper parameters in `hyperparams.py` if necessary.
  * STEP 2. Download and extract DSD100 data as mentioned above at 'data' directory, and run `data.py`.
  * STEP 3. Run `train.py`. 

## Generate seperated vocal audio
  * Prepare a test data (the name of this data should be defined at hyperparams.py) and locate it in 'data' directory and run `eval.py`.

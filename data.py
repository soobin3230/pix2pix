import os
import numpy as np
import librosa
import hyperparams as hp

def _load_data():
    mixtures, vocals = list(), list()
    for path, subdirs, files in os.walk('./data/DSD100/Mixtures/Dev'):
        for name in [f for f in files if f.endswith(".wav")]:
            mixtures.append(os.path.join(path, name))

    for path, subdirs, files in os.walk('./data/DSD100/Sources/Dev'):
        for subdir in subdirs:
            vocal = os.path.join(os.path.join(path, subdir), "vocals.wav")
            vocals.append(vocal)

    num_wavs = len(mixtures)

    return mixtures, vocals, num_wavs

def _get_rawwave(_input):
    return librosa.load(_input, sr=hp.sample_rate)

def _rawwave_to_spectrogram(_input):
    return librosa.stft(y=_input, n_fft=hp.fft_size, hop_length=hp.hop_length, win_length=hp.window_size)

def _spectrogram_to__rawwave(_input):
    return librosa.istft(_input, hop_length=hp.hop_length, win_length=hp.window_size)

def _get_spectrogram():

    rawwave_size = hp.duration * hp.sample_rate

    if hp.is_training:
        m, v, n = _load_data()

        arrays = []
        arrays_2 = []
        for i, j in zip(m, v):
            mixture = _get_rawwave(i)[0]
            vocal = _get_rawwave(j)[0]
            data_length = len(mixture)
            num_spectrograms = data_length // rawwave_size

            for k in range(num_spectrograms):
                arrays.append(_rawwave_to_spectrogram(mixture[k*rawwave_size : (k+1)*rawwave_size]))
                arrays_2.append(_rawwave_to_spectrogram(vocal[k*rawwave_size : (k+1)*rawwave_size]))

        print("start save files...")
        np.save(hp.mixture_data, np.expand_dims(np.stack(arrays), axis=-1))
        np.save(hp.vocal_data, np.expand_dims(np.stack(arrays_2), axis=-1))
        print("save done!")

    else:
        raw_wave = _get_rawwave(hp.eval_wav)[0]
        data_length = len(raw_wave)
        num_spectrograms = data_length // rawwave_size
        arrays = []
        for k in range(num_spectrograms):
            arrays.append(_rawwave_to_spectrogram(raw_wave[k * rawwave_size: (k + 1) * rawwave_size]))

        return np.expand_dims(np.stack(arrays), axis=-1)


if __name__ == '__main__':
    _get_spectrogram()
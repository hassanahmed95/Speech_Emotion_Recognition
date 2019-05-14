import numpy as np
import scipy.io.wavfile as wav
from speechpy.feature import mfcc
import soundfile as sf
import os

mean_signal_length = 150243
Data_Source = "/home/hassan/Hassaan_Home/ML_Datasets/SER_Dataset/Complete_Prepared_Dataset/Training_Data"



def min_max_scalling(data):
    normalzied_list = []
    for value in data:
        normalzied_list.append((value - np.min(data)) /
                       (np.max(data)-np.min(data)))

    return np.array(normalzied_list)


def get_feature_vector_from_mfcc(file_path: str, mfcc_len: int ):
    # sf, signal = wav.read(file_path)
    signal, fs = sf.read(file_path)
    s_len = len(signal)
    fs = 48000 # the sample rate of most fast signal in the data. . .


    # pad the signals to have same size if lesser than required
    if s_len < mean_signal_length:
        pad_len = mean_signal_length - s_len
        pad_len //= 2
        pad_rem = pad_len % 2
        signal = np.pad(signal, (pad_len, pad_len + pad_rem),
                        'constant', constant_values=0)
    else:
        pad_len = s_len - mean_signal_length
        pad_len //= 2

        signal = signal[pad_len:pad_len + mean_signal_length]
    # print(len(signal))

    mel_coefficients = mfcc(signal, fs, num_cepstral=mfcc_len)

    mel_coefficients = np.ravel(mel_coefficients)
    print(mel_coefficients.shape)

    # print(len(mel_coefficients))
    # exit()

    normalize_feature_vector= min_max_scalling(mel_coefficients)
    # print(len(normalize_feature_vector))
    # exit()

    return normalize_feature_vector


def get_data (data_path=Data_Source, mfcc_len = 70, class_labels=("Angry","Happy","Neutral", "Sad")):
    data = []
    labels = []
    names = []
    os.chdir(data_path)
    for i, directory in enumerate(class_labels):
        os.chdir(directory)
        print(directory)

        for filename in os.listdir('.'):
            print(filename)

            filepath = os.getcwd() + '/' + filename
            feature_vector = get_feature_vector_from_mfcc(file_path=filepath,mfcc_len=mfcc_len)

            data.append(feature_vector)
            labels.append(i)
            names.append(filename)
        os.chdir("..")

    return np.array(data), np.array(labels)


if __name__ == "__main__":
    data, labels = get_data()
    print(len(data))
    print(len(labels))



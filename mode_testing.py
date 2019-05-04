import numpy as np
import scipy.io.wavfile as wav
from speechpy.feature import mfcc
import soundfile as sf
import os
import pickle

mean_signal_length = 45000

def min_max_scalling(data):
    my_list = []
    for value in data:
        my_list.append((value - np.min(data)) /
                       (np.max(data)-np.min(data)))

    return np.array(my_list)

def get_feature_vector_from_mfcc(file_path: str, mfcc_len: int =45 ):
    # sf, signal = wav.read(file_path)
    signal, fs = sf.read(file_path)
    s_len = len(signal)
    if s_len < mean_signal_length:
        # print("I am in the if loop, for the features padding")
        pad_len = mean_signal_length - s_len
        pad_len //= 2
        pad_rem = pad_len % 2
        signal = np.pad(signal, (pad_len, pad_len + pad_rem),
                        'constant', constant_values=0)

    else:
        # print("I am in the else loop")
        pad_len = s_len - mean_signal_length
        pad_len //= 2

        signal = signal[pad_len:pad_len + mean_signal_length]

    mel_coefficients = mfcc(signal, fs, num_cepstral=mfcc_len)
    mel_coefficients = np.ravel(mel_coefficients)

    normalize_feature_vector= min_max_scalling(mel_coefficients)

    return normalize_feature_vector


def testing():

    pickle_in = open("/home/hassan/Hassaan_Home/My_Python_Projects/Speech_Project/My_ML_Models/SVM_Model.pickle", "rb")
    model = pickle.load(pickle_in)
    file = "/home/hassan/Hassaan_Home/Digityfy_Projects/SER_Tech_Stuff/berlin_dataset/Sad/13a07Tc.wav"
    data = get_feature_vector_from_mfcc(file)

    prediction = model.predict([data])[0]

    if prediction == 0:
        print("The prediction is" + " Angry")
    elif prediction ==1:
        print("The prediction is" + " happy")
    elif prediction ==2:
        print("The prediction is" + " Neutral")
    else :
        print("The prediction is" + " Sad")

    # file = open("x_test.txt", "r")
    # reading = file.read()
    # signals = np.array(list(reading))
    # print(len(signals))
    # print(signals)
    # exit()
    # predictions =[]

    # for i in signals:
    #     print(i)
    #     data =  get_feature_vector_from_mfcc(i)
    #     prediction = model.predict([data])[0]
    #     predictions.append(prediction)
    # print(predictions)


if __name__ == "__main__":
    testing()



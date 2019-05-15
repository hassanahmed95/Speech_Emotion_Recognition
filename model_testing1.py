import numpy as np
import scipy.io.wavfile as wav
from speechpy.feature import mfcc
import soundfile as sf
import os
import pickle

file_path = "/home/hassan/Hassaan_Home/ML_Datasets/SER_Dataset/Complete_Prepared_Dataset/Testing_Data"

mean_signal_length = 150243

prediction_labels =[]


def min_max_scalling(data):
    my_list = []
    for value in data:
        my_list.append((value - np.min(data)) /
                       (np.max(data)-np.min(data)))

    return np.array(my_list)


def get_feature_vector_from_mfcc(file_path: str, mfcc_len: int =70 ):
    # sf, signal = wav.read(file_path)
    signal, fs = sf.read(file_path)
    s_len = len(signal)

    fs = 48000

    if s_len < mean_signal_length:
        # print("I am in the if loop, for the features padding")
        pad_len = mean_signal_length - s_len
        pad_rem = pad_len % 2
        pad_len //= 2
        signal = np.pad(signal, (pad_len, pad_len + pad_rem),
                        'constant', constant_values=0)
    else:
        # print("I am in the else loop")
        pad_len = s_len - mean_signal_length
        pad_len //= 2
        signal = signal[pad_len:pad_len + mean_signal_length]

    mel_coefficients = mfcc(signal, fs, num_cepstral= mfcc_len)
    mel_coefficients = np.ravel(mel_coefficients)
    normalize_feature_vector = min_max_scalling(mel_coefficients)
    return normalize_feature_vector

#my that method has been implemented just to perform the testing

def testing():

    pickle_in = open("My_ML_Models/SVM_Model_Updated.pickle", "rb")
    model = pickle.load(pickle_in)
    class_labels = ("Angry", "Happy", "Neutral", "Sad")
    os.chdir(file_path)
    for directory in class_labels:
        os.chdir(directory)
        for filename in os.listdir('.'):
            # print(filename)
            filepath = os.getcwd() + '/' + filename
            data = get_feature_vector_from_mfcc(filepath)
            prediction = model.predict([data])[0]
            prediction_labels.append(prediction)
            # print(prediction)
        os.chdir("..")

    return prediction_labels


def calculating_accuracy():

    with open("/home/hassan/Hassaan_Home/My_Python_Projects/Speech_Emotion/Testing_labels/testing_labels2.txt","rb") as f:
        testing_labels = pickle.load(f)

    predictions = testing()

    print(len(testing_labels))
    print(len(predictions))
    count = 0.0
    correct = 0
    for i in range(len(predictions)):
        count = count + 1
        if testing_labels[i] == predictions[i]:
            correct = correct + 1

    M = correct / count
    Accuray = M * 100
    print("Accuracy is ", Accuray, " percent")
    print("Everything is done. . . . . .")


if __name__ == "__main__":

    calculating_accuracy()



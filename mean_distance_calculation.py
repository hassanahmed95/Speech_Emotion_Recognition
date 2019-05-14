import os
import soundfile as sf
import numpy as np
from shutil import copyfile
import shutil
from speechpy.feature import mfcc
def finding_mean_length():
    root_path = "/home/hassan/Hassaan_Home/ML_Datasets/SER_Dataset/Complete_Prepared_Dataset/test_training"
    signal_data = []
    sample_rate =[]

    all_folders = os.listdir(root_path)
    for i in all_folders:
        print(i)
        current_folder = root_path + "/" + i
        os.chdir(current_folder)
        data = os.listdir(current_folder)
        for wav_file in data:

            signal, fs = sf.read(wav_file)
            signal_data.append(len(signal))
            sample_rate.append(fs)
            # data.append(signal)

    # results = np.array(data)
    # print(results)
    # sample_rate = np.array(sample_rate)
    print(np.mean(signal_data).round(2))
    print(min(sample_rate))


#here im updating the code  . .
#here I am just copying the data from the one folder to another folder


def merging_files():

    root_path = "/home/hassan/Hassaan_Home/ML_Datasets/SER_Dataset/Audio_Emotions/KL"
    copied_path = "/home/hassan/Hassaan_Home/ML_Datasets/SER_Dataset/copied_data/angry_emotion"

    all_files = os.listdir(root_path)
    # all_files = os.listdir(copied_path)
    for i in all_files:
        # os.chdir(copied_path)
        file_name = root_path + "/" + i
        base_name = os.path.basename(file_name)
        # os.rename(base_name, "DC"+ base_name)
        data_split= base_name.split(".")[0][0]
        data_split2 =  base_name.split(".")[0][1]
        if data_split == "a" :#and data_split2 =='a':
            shutil.copy2(file_name, copied_path)


def changing_file_names():
    copied_path = "/home/hassan/Hassaan_Home/ML_Datasets/SER_Dataset/copied_data/angry_emotion"
    all_files = os.listdir(copied_path)
    for i in all_files:
        os.chdir(copied_path)
        file_name = copied_path + "/" + i
        base_name = os.path.basename(file_name)
        print(base_name)
        os.rename(base_name, "KL_angry_" + base_name)


def complex_file_iteration(): #method for the datasets of speeches and songs (large files . . .)
    #that method is for the datasets of the actors (Speech+ songs), in order to sort them out
    root_path = "/home/hassan/Hassaan_Home/ML_Datasets/SER_Dataset/Audio_Speech_Actors_01-24"
    angry_path=  "/home/hassan/Hassaan_Home/ML_Datasets/SER_Dataset/copied_data/angry_emotion"
    happy_path=  "/home/hassan/Hassaan_Home/ML_Datasets/SER_Dataset/copied_data/happy_emotion"
    neutral_path=  "/home/hassan/Hassaan_Home/ML_Datasets/SER_Dataset/copied_data/neutral_emotion"
    sad_path=  "/home/hassan/Hassaan_Home/ML_Datasets/SER_Dataset/copied_data/sad_emotion"
    r = []
    for root, dirs, files in os.walk(root_path):

        for name in files:
            filename = (os.path.join(root, name))
            r.append(filename)

            base_name = os.path.basename(filename)
            print(base_name)
            expression_id = (base_name.split('.')[0][7])

            # shutil.copy2(filename,copied_path)
            print(expression_id)

            if expression_id == str(1) or expression_id == str(2):
                shutil.copy2(filename, neutral_path)
            if expression_id == str(3):
                shutil.copy2(filename, happy_path)
            if expression_id == str(4):
                shutil.copy2(filename, sad_path)
            if expression_id == str(5):
                shutil.copy2(filename, angry_path)

    print("done")


if __name__ == '__main__':
    # merging_files()
    # changing_file_names()
    # complex_file_iteration()
    #that method has been written to get the mean length of audio signal
    # finding_mean_length()

    mean_signal_length= 150243
    file_name= "/home/hassan/Hassaan_Home/ML_Datasets/SER_Dataset/Complete_Prepared_Dataset/Training_Data/03-01-02-01-02-02-05.wav"
    signal, fs = sf.read(file_name)
    s_len = len(signal)

    signal = np.ravel(signal)
    print(signal.shape)
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

    mel_coefficients = mfcc(signal, fs, num_cepstral=70)
    mel_coefficients = np.ravel(mel_coefficients)
    print(mel_coefficients.shape)

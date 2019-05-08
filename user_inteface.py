import tkinter
from tkinter import *
from tkinter import filedialog
import numpy as np
import soundfile as sf
from speechpy.feature import mfcc
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
    fs = 16000
    # print(s_len)

    if s_len < mean_signal_length:
        # print("I am in the if loop, for the features padding")
        pad_len = mean_signal_length - s_len
        pad_len //= 2
        pad_rem = pad_len % 2
        signal = np.pad(signal, (pad_len, pad_len + pad_rem),
                        'constant', constant_values=0)

    else:
        print("I am in the else loop")
        pad_len = s_len - mean_signal_length
        pad_len //= 2

        signal = signal[pad_len:pad_len + mean_signal_length]

    mel_coefficients = mfcc(signal, fs, num_cepstral=mfcc_len)
    mel_coefficients = np.ravel(mel_coefficients)
    normalize_feature_vector= min_max_scalling(mel_coefficients)

    return normalize_feature_vector



def main_interface():
    #it is main interface and it has been declared as global, so that
    #it can accessed throughout the file

    global master
    master = Tk()
    master.geometry("500x300")
    master.title("Speech Emotion Recognition . . .")

    my_text = "Speech Emotion Recognition Portal"
    msg = Message(master, text=my_text)
    msg.config(justify=CENTER, font=('times', 25, 'italic'))
    msg.pack()
    main_button = Button(master, text="Click here to get started", bg='black',fg='white', command=file_selection)
    main_button.place(x=160, y=160)
    master.mainloop()

def file_selection():
    global master2
    master2 = Tk()
    master2.geometry("500x300")
    master2.title("New Window")

    select_button = Button(master2, text="Browse a file",width=  18, bg='black', fg='white', command = file_procssing)
    select_button.place(x=180, y=16)

    quit_button = Button(master2, text="Quit Application ", width=18, command=exit,bg='black', fg='white')
    quit_button.place(x=180, y=220)

    master.withdraw()
    master2.mainloop()

def file_procssing():

    file1 = filedialog.askopenfilename()
    if not file1 :
        mainloop()
    # print(file1)

    extension = file1.split(".")[1]
    print(extension)
    print(type(extension))

    if extension != "wav":
        print("lala a gyea haan")
        msg = Message(master2, text="Select only .wav file")
        msg.config(justify=CENTER, width=400, font=('times', 15, 'italic'))
        msg.place(x=170, y=120)
        mainloop()

    data = get_feature_vector_from_mfcc(file1)

    pickle_in = open("/home/hassan/Hassaan_Home/My_Python_Projects/Speech_Project/My_ML_Models/SVM_Model.pickle", "rb")
    model = pickle.load(pickle_in)

    prediction = model.predict([data])[0]
    print("Hello word")
    print(prediction)

    if prediction == 0:   
        output = "Predicted Emotion :" + "Angry"+"  "
    elif prediction == 1:
        output = "Predicted Emotion :" + "happy "+"  "
    elif prediction == 2:
        output = "Predicted Emotion :" + "Neutral"+"    "
    else:
        output = "Predicted Emotion :" + " Sad   "+"     "

    msg = Message(master2, text=output)
    msg.config(justify=CENTER, width=400, font=('times', 15, 'italic'))
    msg.place(x=150, y=120)


if __name__ =='__main__':

    main_interface()

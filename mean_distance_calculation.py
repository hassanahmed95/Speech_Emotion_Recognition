import os
import soundfile as sf
import numpy as np

root_path = "/home/hassan/Hassaan_Home/Digityfy_Projects/SER_Tech_Stuff/berlin_dataset"
results = []

all_folders = os.listdir(root_path)

for i in all_folders:
    current_folder = root_path + "/" + i
    os.chdir(current_folder)
    data = os.listdir(current_folder)
    for wav_file in data:
        signal, fs = sf.read(wav_file)
        results.append(len(signal))


results = np.array(results)
print(np.mean(results).round(2))
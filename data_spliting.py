import numpy as np
from sklearn.model_selection import train_test_split
from Features_Extraction import get_data


data_set_path= "/home/hassan/Hassaan_Home/Digityfy_Projects/SER_Tech_Stuff/test"
class_labels=("Angry", "Happy", "Neutral", "Sad")
print(type(class_labels))


def extract_data():
    data, labels = get_data()
    x_train, x_test, y_train, y_test = train_test_split(
        data,
        labels,
        test_size=0.2,
        random_state=42)

    # print(len(class_labels))
    return np.array(x_train), np.array(x_test), np.array(y_train), np.array(
        y_test), len(class_labels)

import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from data_spliting import extract_data
from sklearn.svm import SVC


def model_training():

    # model = RandomForestClassifier(n_estimators=30)
    clf = SVC(C=170.0, kernel='linear', degree=3, gamma='auto', coef0=0.0,
                shrinking=True, probability=True, tol=0.001, cache_size=200,
                class_weight=None, verbose=False, max_iter=-1,
                decision_function_shape='ovr',
                random_state=80)

    # clf = MLPClassifier(verbose=True,
    #                        hidden_layer_sizes=(512,), batch_size=32)

    x_train, x_test, y_train, y_test, _ = extract_data()

    print("Data Reading has been done . .")

    print("model training has been started. . . .")
    clf.fit(x_train , y_train)
    file_name = "/home/hassan/Hassaan_Home/My_Python_Projects/Speech_Project/My_ML_Models/SVM_Model.pickle"
    pickle_out = open(file_name, "wb")
    pickle.dump(clf, pickle_out)
    pickle_out.close()
    print("Trainig done, and model has been saved. . .")


if __name__ == "__main__":
    print("Hello word . . ..")
    model_training()



import numpy as np
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt
import pickle

class JST:
    __data = []
    __x = []
    __y = []
    __settings = {}

    def __init__(self,**settings):
        for key in settings:
            self.__settings[key] = settings[key]

    def __loadDataset(self):
        data = np.genfromtxt(self.__settings["Dataset Path"], delimiter=',')
        print("Load Dataset...........Done!")
        return data

    def __predictBy3(self, data):
        i = 0
        while i < (len(data) - 3):
            self.__x.append([float(data[i, 1]), float(data[i + 1, 1]), float(data[i + 2, 1])])
            self.__y.append(float(data[i + 3, 1]))
            i += 1

    def __predictBy5(self,data):
        i = 0
        while i < (len(data) - 5):
            self.__x.append([float(data[i, 1]), float(data[i + 1, 1]), float(data[i + 2, 1]), float(data[i + 3, 1]),
                      float(data[i + 4, 1])])
            self.__y.append(float(data[i + 5, 1]))
            i += 1

    def __train(self):
        if self.__settings["Load Model"]:
            clf = self.__loadModel()
        else:
            clf = MLPRegressor(activation="identity", solver='sgd', alpha=0.0001, hidden_layer_sizes=(99, 1),
                               verbose=self.__settings["Learning Verbose"],
                               batch_size="auto", max_iter=100, learning_rate='adaptive', learning_rate_init=0.0000001,
                               warm_start=True)
            for _ in range(20):
                clf.fit(self.__x, self.__y)
            print("Model Train............Done!")
        return clf

    def __showPlot(self,x_train,y_train,y_predict):
        if self.__settings["Show Plot"]:
            plt.title(self.__settings["Dataset Path"])
            plt.plot(x_train, y_train, label="Actual Movement")
            if self.__settings["Predict By"] == 5:
                plt.plot(x_train[self.__settings["Predict By"]:], y_predict, label="Prediction Movement")
            else:
                plt.plot(x_train[3:], y_predict, label="Prediction Data")
            plt.legend()
            plt.show()

    def __saveModel(self,model):
        if self.__settings["Save Model"]:
            pickle.dump(model, open(self.__settings["Model Filename"], 'wb'))

    def __loadModel(self):
        clf = pickle.load(open(self.__settings["Model Filename"],'rb'))
        return clf

    def run(self):
        self.__data = self.__loadDataset()
        x_train = self.__data[:, 0]
        y_train = self.__data[:, 1]

        # preProcessing
        if self.__settings["Predict By"]==5:
            self.__predictBy5(self.__data)
        else:
            self.__predictBy3(self.__data)

        clf = self.__train()

        y_predict = np.array(clf.predict(self.__x))

        self.__saveModel(clf)
        self.__showPlot(x_train,y_train,y_predict)
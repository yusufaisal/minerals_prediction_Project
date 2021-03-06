import numpy as np
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt
import random as rand
import pickle

class JST:
    __clf = None
    __data = []
    __x = []
    __y = []
    __settings = {}

    def __init__(self,**settings):
        for key in settings:
            self.__settings[key] = settings[key]
        self.__clf = None
        self.__data = []
        self.__x = []
        self.__y = []

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

    def __predictBy7(self,data):
        i = 0
        while i < (len(data) - 7):
            self.__x.append([float(data[i, 1]), float(data[i + 1, 1]), float(data[i + 2, 1]), float(data[i + 3, 1]),
                      float(data[i + 4, 1]),float(data[i + 5, 1]),float(data[i + 6, 1])])
            self.__y.append(float(data[i + 7, 1]))
            i += 1

    def __train(self):
        if self.__settings["Load Model"]:
            clf = self.__loadModel()
        else:
            clf = MLPRegressor(activation="identity", solver='sgd', alpha=0.0001, hidden_layer_sizes=(99, 10),
                               verbose=self.__settings["Learning Verbose"],
                               batch_size="auto", max_iter=100, learning_rate='adaptive', learning_rate_init=0.0000001,
                               warm_start=True)
            for _ in range(40):
                clf.fit(self.__x, self.__y)
            print("Model Train............Done!")
        return clf

    def __showPlot(self,x_train,y_train,y_predict):
        if self.__settings["Show Plot"]:
            plt.title(self.__settings["Dataset Path"])
            plt.plot(x_train, y_train, label="Actual Movement")
            plt.plot(x_train[self.__settings["Predict By"]:], y_predict, label="Prediction Movement")
            plt.legend()
            plt.show()

    def __saveModel(self,model):
        if self.__settings["Save Model"]:
            pickle.dump(model, open(self.__settings["Model Filename"], 'wb'))

    def __loadModel(self):
        clf = pickle.load(open(self.__settings["Model Filename"],'rb'))
        return clf

    def positif(self,x):
        if x <0:
            return x*-1
        else : return x

    def __MAPE(self,y_train,y_predict):
        sum = 0
        for i in range(len(y_predict)):
            x = y_train[i+self.__settings["Predict By"]] - y_predict[i]
            sum += (self.positif(x))/y_train[i+self.__settings["Predict By"]]

        result = sum/len(y_predict)*100
        return result

    def predict(self,End):
        data = self.__data
        x_test = []
        values = []
        years = []
        i = len(data)-(self.__settings["Predict By"]+1)
        n = 2012
        while n < (End):
            if x_test == []:
                x_test.append([[float(data[i][1]),
                                float(data[i + 1][1]),
                                float(data[i + 2][1]),
                                float(data[i + 3][1]),
                                float(data[i + 4][1]),
                                float(data[i + 5][1]),
                                float(data[i + 6][1])]])
                i += 1
            else:
                x_test.append([[x_test[len(x_test) - 1][0][1],
                                x_test[len(x_test) - 1][0][2],
                                x_test[len(x_test) - 1][0][3],
                                x_test[len(x_test) - 1][0][4],
                                x_test[len(x_test) - 1][0][5],
                                x_test[len(x_test) - 1][0][6],
                                y_test[0]]])
            n += 1
            years.append(n)
            qwerty = x_test[len(x_test) - 1]
            y_test = self.__clf.predict(qwerty)
            values.append(y_test[0])

        return years,values

    def run(self):
        self.__data = self.__loadDataset()
        x_train = self.__data[:, 0]
        y_train = self.__data[:, 1]

        # preProcessing
        if self.__settings["Predict By"]==5:
            self.__predictBy5(self.__data)
        elif self.__settings["Predict By"]==7:
            self.__predictBy7(self.__data)
        else:
            self.__predictBy3(self.__data)

        self.__clf = self.__train()
        print("Loss :",self.__clf.loss_)

        y_predict = np.array(self.__clf.predict(self.__x))
        print("MAPE :",self.__MAPE(y_train,y_predict))

        print()
        self.__saveModel(self.__clf)
        self.__showPlot(x_train,y_train,y_predict)
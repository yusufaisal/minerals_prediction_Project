import numpy as np
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt

data = np.genfromtxt('../Dataset/coal-consumption-mtoe.csv',delimiter=',')
x_train = data[:,0]
y_train = data[:,1]

i=0
x = []
y = []
while i<(len(data)-3):
    x.append([float(data[i,1]),float(data[i+1,1]),float(data[i+2,1])])
    y.append(float(data[i+3,1]))
    i += 1

clf = MLPRegressor(activation="identity",solver='sgd',alpha = 1, hidden_layer_sizes = (30, 1), verbose = True,
                   batch_size = "auto",max_iter=100,learning_rate='adaptive',learning_rate_init=0.000001)

print(clf.fit(x, y))
# xx = np.array([0.1,0.3,0.1])
# xx.forma(1,-1)
y_predict = np.array(clf.predict(x))
print(y_predict)
plt.plot(x_train,y_train)
plt.plot(x_train[3:],y_predict)
plt.show()
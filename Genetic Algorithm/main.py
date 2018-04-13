# from sklearn.neural_network import MLPClassifier
from class_GA import GeneticAlgorithm


#### GA Setting ####
GAsettings = {}
GAsettings["Populations"] = 8
GAsettings["Generations"] = 12
GAsettings["Max Number"] = 99
GAsettings["Kromosom"] = 12
GAsettings["Crossover"] = True
GAsettings["Mutation"] = True
GAsettings["Crossover Probability"] = 0.6
GAsettings["Mutation Probability"] = 0.8
GAsettings["Replacement Strategy"] = "Steadystate"  # steadystate or elitism

if __name__=="__main__":
    # X = [[0., 0.], [1., 1.]]
    # y = [0, 1]
    # clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
    #                     hidden_layer_sizes=(5, 2), random_state=1)
    #
    # # print clf.fit(X, y)
    # print clf.predict([[2., 2.], [-1., -2.]])

    ga = GeneticAlgorithm(**GAsettings)
    ga.run()
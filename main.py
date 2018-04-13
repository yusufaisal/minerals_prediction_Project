# from sklearn.neural_network import MLPClassifier
from class_GA import GeneticAlgorithm
from class_JST import JST

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

#### JST Setting ####
JSTSettings = {
    "Dataset Path"      : "Dataset/oil-consumption-tonnes.csv",
    "Model Filename"    : "JST Model/oil_consumption_model.sav",
    "Load Model"        : False,
    "Save Model"        : True,
    "Learning Verbose"  : True,
    "Predict By"        : 5, # 5 or 3(default)
    "Show Plot"         : True
}

if __name__=="__main__":
    jst = JST(**JSTSettings)
    jst.run()

    ga = GeneticAlgorithm(**GAsettings)
    ga.run()

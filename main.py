# from sklearn.neural_network import MLPClassifier
from class_GA import GeneticAlgorithm
from class_JST import JST

#### GA Setting ####
GAsettings = {
    "Populations"   : 8,
    "Generations"   : 12,
    "Max Number"    : 99,
    "Kromosom"      : 12,
    "Crossover"     : True,
    "Mutation"      : True,
    "Crossover Probability" : 0.6,
    "Mutation Probability"  : 0.8
    # "Replacement Strategy"  : "Steadystate"  # steadystate or elitism
}
#### JST Setting ####
JSTSettings = {
    "Dataset Path"      : "Dataset/oil-consumption-tonnes.csv",
    "Model Filename"    : "JST Model/oil_consumption_model.sav",
    "Load Model"        : False,
    "Save Model"        : True,
    "Learning Verbose"  : True,
    "Predict By"        : 5,  #5 or 3(default)
    "Show Plot"         : True
}

if __name__=="__main__":
    jst = JST(**JSTSettings)
    jst.run()

    ga = GeneticAlgorithm(**GAsettings)
    ga.run()

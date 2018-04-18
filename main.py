from class_GA import GeneticAlgorithm
from class_JST import JST
from class_FIS import Fuzzy

#### GA Setting ####
GAsettings = {
    "Populations"   : 8,
    "Generations"   : 12,
    "Max Number"    : 99,
    "Crossover"     : True,
    "Mutation"      : True,
    "Crossover Probability" : 0.6,
    "Mutation Probability"  : 0.8
    # "Kromosom"      : 12,
    # "Replacement Strategy"  : "Steadystate"  # steadystate or elitism
}

#### JST Setting ####
JSTSettings = {
    "Dataset Path"      : "Dataset/oil-consumption-tonnes.csv",
    "Model Filename"    : "JST Model/oil_consumption_model_By5.sav",
    "Load Model"        : True,
    "Save Model"        : False,
    "Learning Verbose"  : True,
    "Predict By"        : 5,  #5 or 3(default)
    "Show Plot"         : True
}

if __name__=="__main__":

    jst = JST(**JSTSettings)
    jst.run()

    # ga = GeneticAlgorithm(**GAsettings)
    # ga.run()

    # FIZ = Fuzzy([[8, 33], [8, 28], [35, 55], [5, 52], [[[0, 0], [1, 0]], [[0, 1], [1, 0]]]])
    # FIZ.run(11,20,20)
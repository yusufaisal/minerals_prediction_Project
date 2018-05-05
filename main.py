from method.class_JST import JST
from method.class_GA import GeneticAlgorithm

#### GA Setting ####
GAsettings = {
    "Populations"   : 200,
    "Generations"   : 1,
    "Crossover"     : True,
    "Mutation"      : False,
    "Crossover Probability" : 0.6,
    "Mutation Probability"  : 0.8,
    "Production file"   : "Dataset/oil-production-tonnes.csv",
    "Consumption file"  : "Dataset/oil-consumption-tonnes.csv",
    "Class file"    : "Dataset/oil_class.csv",
    # "Kromosom"      : 12,
    # "Replacement Strategy"  : "Steadystate"  # steadystate or elitism
}

#### JST Setting ####
JSTSettings = {
    "Dataset Path"      : "Dataset/coal-consumption-mtoe.csv",
    "Model Filename"    : "JST Model/coal_consumption_model_By7.sav",
    "Load Model"        : False,
    "Save Model"        : False,
    "Learning Verbose"  : True,
    "Predict By"        : 7,  #7, 5, or 3(default)
    "Show Plot"         : True
}

if __name__=="__main__":
    # jst = JST(**JSTSettings)
    # jst.run()

    ga = GeneticAlgorithm(**GAsettings)
    # print(ga.fitness([[2, 63, 63, 73, 90, 94], [7, 11, 39, 63, 84, 88], [0.2364993194401892, 0.3628417958341108, 0.8852613625610533], [[0, 1, 2, 0], [2, 2, 2, 2], [2, 1, 2, 2], [1, 1, 1, 2]]]))
    ga.run()
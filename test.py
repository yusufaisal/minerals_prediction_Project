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
    arr = [[3, 39, 58, 60, 60, 74], [11, 15, 38, 39, 53, 81], [0.04909013365414383, 0.5393633962860299, 0.8613001492527057], [[0, 1, 2, 2], [1, 0, 1, 0], [1, 0, 2, 1], [0, 2, 2, 0]]]
    print(ga.fitness(arr))
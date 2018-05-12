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
    for i in range(5):
        ga = GeneticAlgorithm(**GAsettings)

        param = [[22, 27, 39, 39, 59, 62], [11, 44, 57, 62, 72, 90],
                 [0.3468746509230485, 0.3685844563743308, 0.7699144850522315],
                 [[2, 1, 2, 0], [1, 2, 1, 0], [1, 2, 2, 1], [0, 0, 2, 2]]]
        print("\nAccuracy:", ga.test(param))

        param = [[1, 5, 14, 51, 72, 73], [34, 41, 45, 58, 97, 98],
                 [0.22341237600782782, 0.45571745504685135, 0.973699768208969],
                 [[2, 0, 0, 0], [1, 0, 1, 1], [0, 0, 2, 0], [1, 0, 0, 1]]]
        # param = [[11, 31, 46, 51, 67, 88], [2, 5, 30, 44, 66, 69], [0.404794601951741, 0.4574010441256189, 0.6502489101902879], [[1, 0, 1, 2], [2, 0, 2, 1], [1, 2, 2, 2], [1, 0, 1, 1]]]

        print("\nAccuracy:", ga.test(param))
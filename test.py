from method.class_JST import JST
from method.class_GA import GeneticAlgorithm

#### GA Setting ####
uhuy = "oil"
GAsettings = {
    "Data"          : uhuy,
    "Populations"   : 6000,
    "Generations"   : 10,
    "Crossover"     : True,
    "Mutation"      : False,
    "Crossover Probability" : 0.6,
    "Mutation Probability"  : 0.8,
    "Production file"   : "Dataset/"+uhuy+"-production.csv",
    "Consumption file"  : "Dataset/"+uhuy+"-consumption.csv",
    "Class file"        : "Dataset/"+uhuy+"_class.csv",
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
    # for i in range(5):
    ga = GeneticAlgorithm(**GAsettings)

    if uhuy=="coal":
        # coal
        param = [[37, 42, 53, 62, 76, 97], [5, 16, 36, 80, 81, 92], [0.22041392035373764, 0.2727645531271008, 0.4730778821782369], [[0, 1, 2, 1], [2, 2, 1, 2], [0, 1, 0, 2], [1, 1, 0, 2]]]
    elif uhuy=="gas":
        # gas
        param = [[12, 22, 27, 64, 70, 90], [12, 21, 30, 36, 59, 84], [0.01961226236542557, 0.0265507352917832, 0.3958541452293558], [[0, 0, 0, 1], [2, 2, 2, 2], [2, 2, 2, 1], [1, 1, 1, 0]]]
    elif uhuy=="oil":
        # oil
        param = [[7, 8, 26, 74, 78, 92], [4, 40, 61, 62, 74, 76], [0.17733381252824376, 0.3162646281869512, 0.8231895034572642], [[0, 1, 0, 1], [0, 2, 2, 0], [1, 1, 0, 0], [2, 0, 1, 2]]]

    print("Accuracy:", ga.test(param))

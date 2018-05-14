from method.class_JST import JST
from method.class_GA import GeneticAlgorithm

#### GA Setting ####
uhuy = "oil"
GAsettings = {
    "Data"          : uhuy,
    "Populations"   : 300,
    "Generations"   : 10,
    "Crossover"     : True,
    "Mutation"      : False,
    "Crossover Probability" : 0.6,
    "Mutation Probability"  : 0.8,
    "Production file"   : "Dataset/"+uhuy+"-production.csv",
    "Consumption file"  : "Dataset/"+uhuy+"-consumption.csv",
    "Class file"        : "Dataset/"+uhuy+"_class.csv",
    # "Replacement Strategy"  : "Steadystate"  # steadystate or elitism
}

#### JST Setting ####
JSTSettings = {
    "Dataset Path"      : "Dataset/coal-consumption.csv",
    "Model Filename"    : "JST Model/coal_consumption_model_By7.sav",
    "Load Model"        : False,
    "Save Model"        : False,
    "Learning Verbose"  : True,
    "Predict By"        : 7,  #7, 5, or 3(default)
    "Show Plot"         : False
}

if __name__=="__main__":
    jst = JST(**JSTSettings)
    jst.run()
    # jst.predict()

    # ga = GeneticAlgorithm(**GAsettings)
    # ga.run()
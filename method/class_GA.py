# from __future__ import division
from method.class_FIS import Fuzzy
from pprint import pprint
import random
import numpy as np
import pandas as pd

class GeneticAlgorithm(object):
    __settings = {}
    productions = []
    consumptions = []
    classification = []

    def __init__(self,**settings):
        for field in settings:
            self.__settings[field] = settings[field]
        self.productions = np.loadtxt(self.__settings["Production file"],delimiter=',')
        self.consumptions = np.loadtxt(self.__settings["Consumption file"],delimiter=',')
        self.classification = pd.read_csv(self.__settings["Class file"],delimiter=',')

    def __mutation(self,c1,c2):
        if self.__settings["Mutation"]:
            for _ in range(2):
                rand = random.uniform(0, 1)
                x = random.randint(0,1)
                y = random.randint(0, 1)
                z = random.randint(0, 1)
                if rand >= self.__settings["Mutation Probability"]:
                    if c1[x][y][z] == 0:
                        c1[x][y][z] = 1
                    else:
                        c1[x][y][z] = 0

                rand = random.uniform(0, 1)
                x = random.randint(0, 1)
                y = random.randint(0, 1)
                z = random.randint(0, 1)
                if rand >= self.__settings["Mutation Probability"]:
                    if c2[x][y][z] == 0:
                        c2[x][y][z] = 1
                    else:
                        c2[x][y][z]= 0

    def __populations(self,nPop):
        pop = []
        dim  = 4
        for j in range(nPop):
            fk_production = [random.randint(1,99) for i in range(6)]
            fk_consumption = [random.randint(1,99) for i in range(6)]
            fk_out = [random.uniform(0,1) for i in range(3)]
            rule = [[random.randint(0,2) for j in range(dim)] for i in range(dim)]

            pop.append([sorted(fk_production),sorted(fk_consumption),sorted(fk_out),rule])
        return pop

    def __crossover(self,c1,c2,rand):
        if self.__settings["Crossover"]:
            if rand >= self.__settings["Crossover Probability"]:
                titik = random.randint(0,2)
                c1[titik],c2[titik] = c2[titik],c1[titik]

                c1[3][0], c2[3][0] = c2[3][0], c1[3][0]

    def __fitness(self,arr):
        count =0
        result = []
        for i in range(len(self.productions)):
            fz = Fuzzy(arr)
            result.append(fz.run(self.productions[i][1],self.consumptions[i][1]))
            if  result[i][1] == self.classification['Aktual'][i]:
                count+=1
            del fz

        accuracy = (count/len(self.classification['Aktual']))*100
        # if accuracy>80:
        #     print(arr,accuracy)

        return accuracy

    def test(self,param):
        count =0
        fz = Fuzzy(param)
        result = []
        # pprint(fz.setting)

        for i in range(len(self.productions)):
            result.append(fz.run(self.productions[i][1],self.consumptions[i][1],i))
            if (result[i][1] == self.classification['Aktual'][i]):
                count+=1

        result = np.array(result)
        df = self.classification.assign(Predict=result[:,1],CrispOut=result[:,0])
        df['Production'] = self.productions[:,1]
        df['Consumption'] = self.consumptions[:,1]
        accuracy = (count/len(self.classification['Aktual']))*100
        del fz

        df.to_csv("Dataset/"+self.__settings["Data"]+"_result_"+str(accuracy)+".csv",index=False)
        print(df)
        return accuracy

    def run(self):
        populations = self.__populations(self.__settings["Populations"])
        populations.append([[9, 20, 41, 66, 82, 94], [41, 43, 72, 78, 87, 93], [0.21667536156936074, 0.27684040127537657, 0.6390034729226843], [[1, 1, 0, 1], [1, 2, 0, 0], [0, 1, 2, 1], [1, 0, 2, 2]]])
        # print(populations)
        for _ in range(self.__settings["Generations"]):
            child = []
            fitness = []

            for i in range(int(self.__settings["Populations"]/2)):
                parent1 = random.randint(0, self.__settings["Populations"] - 1)
                parent2 = random.randint(0, self.__settings["Populations"] - 1)
                child1 = populations[parent1][:]
                child2 = populations[parent2][:]
                self.__crossover(child1,child2,random.uniform(0,1))
                self.__mutation(child1, child2)
                child.append(child1)
                child.append(child2)

            gab = populations + child
            for j in range(len(gab)):
                fitness.append(self.__fitness(gab[j]))

            steadyState = sorted(range(len(fitness)),key=lambda x: fitness[x], reverse=True)
            # for i in range(len(steadyState)):
            #     print(gab[steadyState[i]],fitness[steadyState[i]])
            # print(steadyState)

            populations = []
            for j in range(self.__settings["Populations"]):
                populations.append(gab[steadyState[j]])
            print("Parameter: ",gab[steadyState[0]])
            print("Accuracy: ",fitness[steadyState[0]])
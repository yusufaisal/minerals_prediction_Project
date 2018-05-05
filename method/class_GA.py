from __future__ import division
from method.class_FIS import Fuzzy
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

    def fitness(self,arr):
        count =0
        fz = Fuzzy(arr)

        for i in range(len(self.productions)):
            result = fz.run(self.productions[i][1],self.consumptions[i][1])
            print(i,result , self.classification['Status'][i])
            if  result == self.classification['Status'][i]:
                count+=1
        # print(len(self.classification['Status']))
        accuracy = (count/len(self.classification['Status']))*100
        if accuracy>80:
            print(arr,accuracy)
        return accuracy

    def run(self):
        populations = self.__populations(self.__settings["Populations"])

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
                fitness.append(self.fitness(gab[j]))

            steadyState = sorted(range(len(fitness)),key=lambda x: fitness[x], reverse=True)
            print(steadyState)

            populations = []
            for j in range(self.__settings["Populations"]):
                print(gab[steadyState[j]])
                populations.append(gab[steadyState[j]])
            print("Parameter: ",gab[steadyState[0]])
            print("accuracy: ",fitness[steadyState[0]])
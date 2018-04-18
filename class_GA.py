from __future__ import division
import random
import numpy as np


class GeneticAlgorithm(object):
    __settings = {}

    def __init__(self,**settings):
        for field in settings:
            self.__settings[field] = settings[field]

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
        dim  = 2
        for j in range(nPop):
            fk_oil = [random.randint(0,self.__settings["Max Number"]) for i in range(2)]
            fk_gas = [random.randint(0,self.__settings["Max Number"]) for i in range(2)]
            fk_coal = [random.randint(0,self.__settings["Max Number"]) for i in range(2)]
            fk_out = [random.randint(0,self.__settings["Max Number"]) for i in range(2)]
            rule = [[[random.randint(0,1) for k in range(dim)] for j in range(dim)] for i in range(dim)]

            pop.append([sorted(fk_oil),sorted(fk_gas),sorted(fk_coal),sorted(fk_out),rule])
        return pop

    def __crossover(self,c1,c2,rand):
        if self.__settings["Crossover"]:
            if rand >= self.__settings["Crossover Probability"]:
                titik = random.randint(0,3)
                c1[titik],c2[titik] = c2[titik],c1[titik]

                c1[4][0], c2[4][0] = c2[4][0], c1[4][0]

    def __fitness(self,arr):

        return None

    def run(self):
        populations = self.__populations(self.__settings["Populations"])
        print(populations)
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

                # print fitness
                steadyState = sorted(range(len(fitness)), key=lambda k: fitness[k], reverse=True)
                # print steadyState

                pop = []
                for j in range(self.__settings["Populations"]):
                    pop.append(gab[steadyState[j]])
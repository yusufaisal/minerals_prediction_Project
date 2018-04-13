from __future__ import division
import random

class GeneticAlgorithm(object):
    __settings = {}

    def __init__(self,**settings):
        for field in settings:
            self.__settings[field] = settings[field]

    def __mutation(self,c1,c2):
        if self.__settings["Mutation"]:
            for _ in range(2):
                rand = random.uniform(0, 1)
                titik = int(round(random.uniform(0, self.__settings["Kromosom"] - 1)))
                if rand >= self.__settings["Mutation Probability"]:
                    if c1[titik] == 0:
                        c1[titik] = 1
                    else:
                        c1[titik] = 0

                rand = random.uniform(0, 1)
                titik = int(round(random.uniform(0, self.__settings["Kromosom"] - 1)))
                if rand >= self.__settings["Mutation Probability"]:
                    if c2[titik] == 0:
                        c2[titik] = 1
                    else:
                        c2[titik] = 0

    def __populations(self,nKrom,nPop):
        pop=[]
        for j in range(nPop):
            x = [int(round(random.randint(0, 1))) for i in range(nKrom)]
            while self.__check(x,nKrom):
                x=[int(round(random.randint(0, 1))) for i in range(nKrom)]
            pop.append(x)

        return pop

    def __check(self,arr,nKrom):
        sum =0
        for i in range(int(nKrom/2)):
            sum += arr[i] * pow(2, i)
        if sum >= self.__settings["Max Number"]:
            return True

        for j in range(i,nKrom):
            sum += arr[j] * pow(2, i)
        if sum >= self.__settings["Max Number"]:
            return True

        return False

    def __crossover(self,c1,c2,rand):
        if self.__settings["Crossover"]:
            if rand >= self.__settings["Crossover Probability"]:
                titik = int(round(random.uniform(0, self.__settings["Kromosom"] - 1)))
                for k in range(titik):
                    c1[k], c2[k] = c2[k], c1[k]

    def __fitness(self,arr):

        return None

    def run(self):
        populations = self.__populations(self.__settings["Kromosom"],self.__settings["Populations"])
        # print (populations)
        for _ in range(self.__settings["Generations"]):
            child = []
            fitness = []

            for i in range(int(self.__settings["Populations"]/2)):
                parent1 = random.randint(0, self.__settings["Populations"] - 1)
                parent2 = random.randint(0, self.__settings["Populations"] - 1)

                child1 = populations[parent1][:]
                child2 = populations[parent2][:]

                self.__crossover(child1,child2,random.uniform(0,1))
                self.__mutation(child1, child2,)

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
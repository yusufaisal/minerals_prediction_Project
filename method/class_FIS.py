from pprint import pprint
class Fuzzy:
    setting = {}
    __fk_production = []
    __fk_consumption = []
    __fk_coal = []
    __resultInference = []

    def __init__(self, data):
        self.setting = {
            "fk_production"    : data[0],
            "fk_consumption"    : data[1],
            "fk_out"    : data[2],
            "rule"      : data[3]
        }
    def __delete__(self, instance):
        print("deleted")
        del self.__fk_coal
        del self.__fk_consumption
        del self.__fk_production
        del self.__resultInference
        del self.setting

    def __getResult(self,x,range):
        # ex. data -> [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        # return [[label,value]]
        # print(x)
        if (x<=range[0]) :
            return [[0,1]]
        elif (x > range[0]) and (x < range[1]):
            y1 = -1*(x-range[1])/(range[1]-range[0])
            y2 = (x - range[0]) / (range[1] - range[0])
            return [[0,y1],[1,y2]]
        elif (x>=range[1]) and (x<=range[2]):
            return [[1, 1]]
        elif (x > range[2]) and (x < range[3]):
            y1 = -1*(x-range[3])/(range[3]-range[2])
            y2 = (x - range[2]) / (range[3] - range[2])
            return [[1,y1],[2,y2]]
        elif (x>=range[3]) and (x<=range[4]):
            return [[2, 1]]
        elif (x > range[4]) and (x < range[5]):
            y1 = -1*(x-range[5])/(range[5]-range[4])
            y2 = (x - range[4]) / (range[5] - range[4])
            return [[2,y1],[3,y2]]
        elif (x>=range[5]):
            return [[3, 1]]

    def __fuzzification(self,production,consumption):
        self.__fk_production = self.__getResult(production, self.setting["fk_production"])
        self.__fk_consumption = self.__getResult(consumption, self.setting["fk_consumption"])

    def __inference(self):
        tempResult = []
        self.__resultInference = []
        k = 0

        for idx_production in range(len(self.__fk_production)):
            for idx_consumption in range(len(self.__fk_consumption)):
                    x0 = self.__fk_production[idx_production][0]
                    y0 = self.__fk_consumption[idx_consumption][0]
                    self.__resultInference.append([self.setting["rule"][x0][y0],
                                                   min([self.__fk_production[idx_production][1],
                                                        self.__fk_consumption[idx_consumption][1]])])
        self.__resultInference.sort()
        tempResult.append(self.__resultInference[0])
        for i in range(len(self.__resultInference)):
            if tempResult[k][0] == self.__resultInference[i][0]:
                if tempResult[k][1] < self.__resultInference[i][1]:
                    tempResult[k][1] = self.__resultInference[i][1]
            else:
                k+=1
                tempResult.append(self.__resultInference[i])
        self.__resultInference = tempResult

    def __defuzzification(self,rule):
        A = 0
        B = 0
        for i in range(len(rule)):
            A += rule[i][1] * self.setting["fk_out"][rule[i][0]]
            B += rule[i][1]
        result = A/B
        return result

    def run(self,production,consumption,i):
        self.__fuzzification(production,consumption)
        self.__inference()
        result = self.__defuzzification(self.__resultInference)
        # if i==8:
        #     print(consumption,production)
        #     print(self.__fk_consumption,self.__fk_production)
        #     pprint(self.__resultInference)
        #     print(result)
        del self.__resultInference
        if result <= 0.5:
            return [result,"Aman"]
        else:
            return [result,"Krisis"]
class Fuzzy:
    setting = {}
    oil_input = 0
    __fk_oil = []
    __fk_gas = []
    __fk_coal = []
    __resultInference = []

    def __init__(self, data):
        self.setting = {
            "fk_oil"    : data[0],
            "fk_gas"    : data[1],
            "fk_coal"   : data[2],
            "fk_out"    : data[3],
            "rule"      : data[4]
        }

    def __getResult(self,x,range):
        if x <= range[0]:
            return [[0,1]]
        elif x>= range[1]:
            return [[1,1]]
        else:
            y1 = -1*(x-range[1])/(range[1]-range[0])
            y2 = (x - range[0]) / (range[1] - range[0])

        return [[0,y1],[1,y2]]

    def __fuzzification(self,oil,gas,coal):
        self.__fk_oil = self.__getResult(oil,self.setting["fk_oil"])
        self.__fk_gas = self.__getResult(gas, self.setting["fk_gas"])
        self.__fk_coal = self.__getResult(coal, self.setting["fk_coal"])

    def __inference(self):
        for idx_oil in range(len(self.__fk_oil)):
            for idx_gas in range(len(self.__fk_gas)):
                for idx_coal in range(len(self.__fk_coal)):
                    self.__resultInference.append([self.setting["rule"][self.__fk_oil[idx_oil][0]][self.__fk_gas[idx_gas][0]][self.__fk_coal[idx_coal][0]],
                                                   min([self.__fk_oil[idx_oil][1],self.__fk_gas[idx_gas][1],self.__fk_coal[idx_coal][1]])])
        print(self.__resultInference)

    def __defuzzification(self):
        return None

    def run(self,oil,gas,coal):
        self.__fuzzification(oil,gas,coal)
        self.__inference()
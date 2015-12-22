SiValues = [4.26, 0.38, 1.56, 44]
GaAsValues = [6.85, 2.1, 2.9, 341]


class Compound:
    def __init__(self, params):
        self.y1 = params[0]
        self.y2 = params[1]
        self.y3 = params[2]
        self.delta = params[3]
        self.name = None
        self.setName(params)

    def setName(self,params):
        if params == GaAsValues:
            self.name = "GaAs"
        elif params == SiValues:
            self.name = "Si"
        else:
            self.name = "Unknown"

    def getName(self):
        return self.name

    def setY1(self, y1):
        self.y1 = y1

    def setY2(self, y2):
        self.y2 = y2

    def setY3(self, y3):
        self.y3 = y3

    def setDelta(self, delta):
        self.delta = delta

    def setParams(self, params):
        self.setY1(params[0])
        self.setY2(params[1])
        self.setY3(params[2])
        self.setDelta(params[3])

    def getParams(self):
        return [self.y1, self.y2, self.y3, self.delta]

    def getY1(self):
        return self.y1

    def getY2(self):
        return self.y2

    def getY3(self):
        return self.y3

    def getDelta(self):
        return self.delta

class PotentialWell:



    def __init__(self, direction="x", depth=130, width=1):
        self.direction = None
        self.depth = depth
        self.width = width
        self.nDirection = None
        self.setDirection(direction)


    def setDirection(self, direction):
        self.direction = str.lower(direction)
        if self.direction == "x":
            self.nDirection = 1
        elif self.direction == "y":
            self.nDirection = 2
        elif self.direction == "z":
            self.nDirection = 3

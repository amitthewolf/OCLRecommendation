from Node import Node


class OpNode(Node):

    def __init__(self,string):
        Node.__init__(self,string)
        self.Left = None
        self.Right = None

    def getRight(self):
        return self.Right

    def getLeft(self):
        return self.Left

    def setLeft(self, OtherNode):
        self.Left = OtherNode

    def setRight(self, OtherNode):
        self.Right = OtherNode

    def getType(self):
        return "Operator"








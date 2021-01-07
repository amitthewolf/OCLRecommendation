class Node:

    # ([@4,12:17='forAll',<19>,1:12] ([@2,5:9='forms',<19>,1:5] ) ([@11,31:31='=',<13>,1:31] ([@10,23:30='mainForm',<19>,1:23] ) true))
    # ([@2,13:13='=',<13>,1:13] ([@0,0:11='isUnmarshall',<19>,1:0] ) true)

    def __init__(self,string):
        self.Root = string
        self.type = ""
        # test for type
        self.Left = None
        self.Right = None

    def getType(self):
        return self.type

    def getRight(self):
        return self.Right

    def getLeft(self):
        return self.Left

    def setLeft(self, OtherNode):
        self.Left = OtherNode

    def setRight(self, OtherNode):
        self.Right = OtherNode

    def getString(self):
        return self.Root







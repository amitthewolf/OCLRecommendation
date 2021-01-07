from Node import Node


# ([@4,12:17='forAll',<19>,1:12] ([@2,5:9='forms',<19>,1:5] ) ([@11,31:31='=',<13>,1:31] ([@10,23:30='mainForm',<19>,1:23] ) true))
# ([@2,13:13='=',<13>,1:13] ([@0,0:11='isUnmarshall',<19>,1:0] ) true)

class AST:

    def __init__(self, ASTString):
       self.RootNode = None
       self.AstString = ASTString
       self.ExpressionString = ""
       self.ParseAST(ASTString)
       self.SetExpression()

    def ParseAST(self,ASTString):
        #...
        End1 = ASTString.find(']')
        RootNode = ASTString[1:End1]
        Root = RootNode.split('\'')
        RootString = Root[1]
        self.RootNode = Node(RootString)
        Rest = ASTString[End1+1:]
        End2 = Rest.find(']')
        Left = Rest[:End2].split('\'')
        self.RootNode.setLeft(Node(Left[1]))
        self.RootNode.setRight(Node("true"))


    def SetExpression(self):
        Node = self.RootNode
        if Node.Left != None:
            self.ExpressionString += Node.Left.getString()
        self.ExpressionString += Node.getString()
        if Node.Right != None:
            self.ExpressionString += Node.Right.getString()




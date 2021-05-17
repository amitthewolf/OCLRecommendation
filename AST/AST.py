from RefNode import RefNode
from OpNode import OpNode


# ([@4,12:17='forAll',<19>,1:12] ([@2,5:9='forms',<19>,1:5] ) ([@11,31:31='=',<13>,1:31] ([@10,23:30='mainForm',<19>,1:23] ) true))
# ([@2,13:13='=',<13>,1:13] ([@0,0:11='isUnmarshall',<19>,1:0] ) true)

class AST:

    def __init__(self, ASTString):
        ASTString = ASTString.replace("SEPERATOR", ".")
        ASTString = ASTString.replace("POINTER", "->")
        self.ExpressionString = ""
        self.RootNode = self.ParseASTNew(ASTString)
        self.AstString = ASTString
        if self.ExpressionString != "ERROR!":
            self.SetExpression()
        self.RefNum = len(self.GetReferences())
        self.OpNum = 0

    # def __init__(self, RootString, LeftString = None, RightString= None,RightIsOP= None):
    #     if LeftString==None:
    #         ASTString = RootString.replace("SEPERATOR", ".")
    #         ASTString = RootString.replace("POINTER", "->")
    #         self.ExpressionString = ""
    #         self.RootNode = self.ParseASTNew(ASTString)
    #         self.AstString = ASTString
    #         if self.ExpressionString != "ERROR!":
    #             self.SetExpression()
    #     else:
    #         self.RootNode = OpNode(RootString)
    #         self.RootNode.Left = RefNode(LeftString)
    #         if RightIsOP:
    #             self.RootNode.Right = OpNode(RightString)
    #         else:
    #             self.RootNode.Right = RefNode(RightString)
    #         if self.ExpressionString != "ERROR!":
    #             self.SetExpression()


    # #best Current Version
    def ParseASTNew(self,ASTString):
        try:
            End1 = ASTString.find(']')
            RootString = ASTString[1:End1]
            Split1 = RootString.split('\'')
            RootString = Split1[1]
            RootNode = self.ExtractOpNode(RootString)
            Rest = self.GetRemainder(End1+1,ASTString)
            Layers, EndIndex = self.CheckLayers(Rest)
            if Layers>1:
                RootNode.setLeft(AST(Rest[:EndIndex]))
            else:
                End2 = Rest.find(']')
                Left = Rest[:End2].split('\'')
                RootNode.setLeft(self.ExtractRefNode(Left[1]))
            Rest = Rest[EndIndex:]
            try:
                Layers, EndIndex = self.CheckLayers(Rest)
                if Layers > 1:
                    RootNode.setRight(AST(Rest[:EndIndex]))
                else:
                    End2 = Rest.find(']')
                    Left = Rest[:End2].split('\'')
                    if len(Left)>1:
                        RootNode.setRight(self.ExtractRefNode(Left[1]))
                    else:
                        Ending = Rest.split(' ')
                        LastNode = Ending[len(Ending) - 1]
                        RootNode.setRight(self.ExtractRefNode(LastNode.replace(")", "")))
            except:
                Ending = Rest.split(' ')
                LastNode = Ending[len(Ending) - 1]
                RootNode.setRight(self.ExtractRefNode(LastNode.replace(")", "")))
            return RootNode
        except:
            print(ASTString)
            self.ExpressionString = "ERROR!"
            return None

    #forall test
    # def ParseASTNew(self,ASTString):
    #     try:
    #         End1 = ASTString.find(']')
    #         RootString = ASTString[1:End1]
    #         Split1 = RootString.split('\'')
    #         RootString = Split1[1]
    #         if RootString.find("forAll"):
    #             RootNode = AST("->",RootString.split("RootString")[0],"forAll",True)
    #         RootNode = self.ExtractOpNode(RootString)
    #         Rest = self.GetRemainder(End1+1,ASTString)
    #         Layers, EndIndex = self.CheckLayers(Rest)
    #         if Layers>1:
    #             RootNode.setLeft(AST(Rest[:EndIndex]))
    #         else:
    #             End2 = Rest.find(']')
    #             Left = Rest[:End2].split('\'')
    #             RootNode.setLeft(self.ExtractRefNode(Left[1]))
    #         Rest = Rest[EndIndex:]
    #         try:
    #             Layers, EndIndex = self.CheckLayers(Rest)
    #             if Layers > 1:
    #                 RootNode.setRight(AST(Rest[:EndIndex]))
    #             else:
    #                 End2 = Rest.find(']')
    #                 Left = Rest[:End2].split('\'')
    #                 if len(Left)>1:
    #                     RootNode.setRight(self.ExtractRefNode(Left[1]))
    #                 else:
    #                     Ending = Rest.split(' ')
    #                     LastNode = Ending[len(Ending) - 1]
    #                     RootNode.setRight(self.ExtractRefNode(LastNode.replace(")", "")))
    #         except:
    #             Ending = Rest.split(' ')
    #             LastNode = Ending[len(Ending) - 1]
    #             RootNode.setRight(self.ExtractRefNode(LastNode.replace(")", "")))
    #         return RootNode
    #     except:
    #         print(ASTString)
    #         self.ExpressionString = "ERROR!"
    #         return None

    def ExtractRefNode(self,string):
        NewNode = RefNode(string)
        return NewNode

    def ExtractOpNode(self, string):
        NewNode = OpNode(string)
        return NewNode

    def GetRemainder(self, index ,string):
        return string[index:]

    def CheckLayers(self, string):
        opener = 0
        closer = 0
        endIndex = 0
        for index in range(len(string)):
            char = string[index]
            if char == '(':
                opener += 1
            if char == ')' and opener > 0:
                closer += 1
            if opener == closer and opener > 0:
                endIndex = index
                return closer,endIndex
        return closer, endIndex

    def SetExpression(self):
        Node = self.RootNode
        if Node.Left != None:
            Leftstr = self.GetNodeString(Node.Left)
        self.ExpressionString += Leftstr + " "+Node.getString()
        Rightstr = self.GetNodeString(Node.Right)
        if Rightstr == "":
            if Node.getString()=="<>":
                self.ExpressionString = Leftstr +" "+ Node.getString() +" "+ "''"
            else:
                self.ExpressionString = Node.getString()+"("+Leftstr+")"
        else:
            self.ExpressionString += " "+Rightstr

    def GetNodeString(self, node):
        if isinstance(node, AST):
            return node.ExpressionString
        else:
            return node.getString()

    def GetReferences(self):
        RefList = []
        Node = self.RootNode
        if Node != None and Node.getString().find("->") != -1:
                RefList.append(Node.getString().split("->")[0])
        if Node != None and Node.Left != None:
            if isinstance(Node.Left, AST):
                ToAdd = Node.Left.GetReferences()
                RefList.extend(ToAdd)
            elif Node.Left.getType() == "Reference":
                RefList.append(Node.Left.getString())
            elif Node.Right.getString().find("->") != -1:
                RefList.append(Node.Right.getString().split("->")[0])
        if Node != None and Node.Right != None:
            if isinstance(Node.Right, AST):
                ToAdd = Node.Right.GetReferences()
                RefList.extend(ToAdd)
            elif Node.Right.getType() == "Reference":
                RefList.append(Node.Right.getString())
            elif Node.Right.getString().find("->")!=-1:
                RefList.append(Node.Right.getString().split("->")[0])
        try:
            RefList = [x for x in RefList if not (x.isdigit() or x=="" or x=="''" or x=="NULL" or x[0] == '-' and x[1:].isdigit() )]
        except:
            return RefList
        return RefList

    def GetOperators(self):
        OpList = []
        Node = self.RootNode
        if Node != None and Node.getType() == "Operator":
            if Node.getString().find("->") != -1:
                OpList.append(Node.getString().split("->")[1])
            else:
                OpList.append(Node.getString())
        if Node != None and Node.Left != None:
            if isinstance(Node.Left, AST):
                ToAdd = Node.Left.GetOperators()
                OpList.extend(ToAdd)
        if Node != None and Node.Right != None:
            if isinstance(Node.Right, AST):
                ToAdd = Node.Right.GetOperators()
                OpList.extend(ToAdd)
        try:
            OpList = [x for x in OpList if not (x.isdigit() or x=="" or x=="''" or x=="NULL" or x[0] == '-' and x[1:].isdigit() )]
        except:
            return OpList
        return OpList
import subprocess
from DAO import DAO
from AST import AST

# dao = DAO()
# dao.ChangeDB('OCLTest.db')
# constraints = dao.GetExpressions()
# counter = 1
# ExpressionsTxt = open("ExpressionsNoChanges.txt",'a',encoding="utf-8")
# for Exp in constraints:
#     # ExpressionsTxt.write("@"+str(Exp[0])+"#"+str(Exp[1].replace(" ","").replace("\n","").replace("\t",""))+"\n")
#     ExpressionsTxt.write("@" + str(Exp[0]) + "#" + str(Exp[1]) + "\n")


ast = AST("([@2,13:13='=',<13>,1:13] ([@0,0:11='isUnmarshall',<19>,1:0] ) true)")
print(ast.ExpressionString)


#
#
# for Exp in constraints:
#     Command = "java -jar OCLtoAST.jar " + str(Exp[1].replace(" ",""))
#     Output = subprocess.run(Command, capture_output=True, text=True)
#     print(str(counter)+" - "+Output.stdout)
#     counter += 1
#     dao.AddAST(Exp[0], Output.stdout)
#
# Command = "java -jar OCLtoAST.jar " + "region->size()=0"
# Output = subprocess.run(Command, capture_output=True, text=True)
# print(Output.stdout)

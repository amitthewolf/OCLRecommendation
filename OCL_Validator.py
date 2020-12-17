import subprocess
from DAO import DAO

dao = DAO()
# dao.ChangeDB('OCLTest.db')
constraints = dao.Get_Expressions_For_Validation()
# counter = 1
Validation_Constraints_TXT = open("Constraints_To_Validate.txt",'a',encoding="utf-8")
for Exp in constraints:
    expression = str(Exp[2]).strip('\n')
    expression = expression.strip('\t')
    expression = expression.strip('\r')
    expression = expression.replace('\n',"")
    expression = expression.replace('\t',"")
    expression = expression.replace('\r',"")
    Validation_Constraints_TXT.write("@" + str(Exp[0]) + "#" + str(Exp[3]) + "#" + str(Exp[1])
                                     + "#" + str(Exp[4]) + "#" + expression + "\n")

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
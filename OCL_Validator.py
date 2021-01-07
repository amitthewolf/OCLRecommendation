from DAO import DAO

dao = DAO()


def export_constraints_to_txt():
    constraints = dao.Get_Expressions_For_Validation()
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


def delete_invalid_constraints_from_db():
    invalid_ocl = open('invalidOCL.txt', 'r')
    Lines = invalid_ocl.readlines()
    count = 0
    # Strips the newline character
    for line in Lines:
        constraint_id = line.split(',')[0]
        dao.delete_invalid_constraints(constraint_id)
        print(constraint_id)

delete_invalid_constraints_from_db()
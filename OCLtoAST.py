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


# Single Test
def SingleTests():
    ast = AST("([@8,53:53='>',<15>,1:53] ([@0,0:33='selfSEPERATORemployeePOINTERexists',<19>,1:0] ([@2,35:35='e',<19>,1:35] )) 45)")
    print(ast.ExpressionString)
    print(ast.GetReferences(), "\n")
    ast = AST("([@6,26:28='and',<52>,1:26] ([@2,18:19='<>',<30>,1:18] ([@0,0:16='selfSEPERATORname',<19>,1:0] ) ([@4,21:24='NULL',<19>,1:21] )) ([@10,48:49='<>',<30>,1:48] ([@8,30:46='selfSEPERATORname',<19>,1:30] ) ))")
    print(ast.ExpressionString)
    print(ast.GetReferences(), "\n")
    ast = AST("([@0,0:36='fromActionSEPERATORinputPOINTERforAll',<19>,1:0] ([@2,38:43='IsType',<19>,1:38] ([@4,45:58='ActionInputPin',<19>,1:45] )))")
    print(ast.ExpressionString)
    print(ast.GetReferences(), "\n")
    ast = AST("([@16,40:46='implies',<59>,1:40] ([@12,34:35='<>',<30>,1:34] ([@0,0:18='selfSEPERATORstates',<19>,1:0] ([@4,23:24='s1',<19>,1:23] ) ([@7,27:28='s2',<19>,1:27] )) ([@14,37:38='s2',<19>,1:37] )) ([@20,64:65='<>',<30>,1:64] ([@18,48:62='s1SEPERATORname',<19>,1:48] ) ([@22,67:81='s2SEPERATORname',<19>,1:67] )))")
    print(ast.ExpressionString)
    print(ast.GetReferences(), "\n")
    ast = AST("([@60,177:183='implies',<59>,4:0] ([@6,45:47='and',<52>,1:45] ([@1,1:24='selfSEPERATORoclIsTypeOf',<19>,1:1] ([@3,26:42='AcceptEventAction',<19>,1:26] )) ([@12,53:72='triggerPOINTERforAll',<19>,2:4] ([@19,108:109='or',<71>,2:59] ([@14,74:93='eventSEPERATORIsType',<19>,2:25] ([@16,95:105='ChangeEvent',<19>,2:46] )) ([@52,142:161='eventSEPERATORIsType',<19>,3:29] ([@54,163:171='CallEvent',<19>,3:50] ))))) ([@66,205:205='=',<13>,4:28] ([@62,185:201='outputPOINTERsize',<19>,4:8] ) 0))")
    print(ast.ExpressionString)
    print(ast.GetReferences(), "\n")
    ast = AST("([@12,57:63='implies',<59>,1:57] ([@8,49:50='<>',<30>,1:49] ([@0,0:41='EmployeeSEPERATORallInstancesPOINTERforAll',<19>,1:0] ([@2,43:43='e',<19>,1:43] )) ([@10,52:55='self',<19>,1:52] )) ([@16,80:81='<>',<30>,1:80] ([@14,65:78='eSEPERATORname',<19>,1:65] ) ([@18,83:99='selfSEPERATORname',<19>,1:83] )))")
    print(ast.ExpressionString)
    print(ast.GetReferences(), "\n")
    ast = AST("([@4,12:17='forAll',<19>,1:12] ([@2,5:9='forms',<19>,1:5] ) ([@11,31:31='=',<13>,1:31] ([@10,23:30='mainForm',<19>,1:23] ) true))")
    print(ast.ExpressionString)
    print(ast.GetReferences(), "\n")
    ast = AST("([@6,22:23='<=',<24>,1:22] ([@2,17:17='+',<31>,1:17] ([@0,0:15='selfSEPERATORage',<19>,1:0] ) 10) ([@8,25:70='selfSEPERATORemployerSEPERATORbossSEPERATORage',<19>,1:25] ))")
    print(ast.ExpressionString)
    print(ast.GetReferences(), "\n")
    ast = AST("([@2,17:17='<',<23>,1:17] ([@0,0:15='selfSEPERATORage',<19>,1:0] ) 40)")
    print(ast.ExpressionString)
    print(ast.GetReferences(), "\n")
    ast = AST("([@12,49:51='and',<52>,1:49] ([@8,26:28='and',<52>,1:26] ([@0,0:2='not',<63>,1:0] ([@4,9:22='oclIsUndefined',<19>,1:9] )) ([@10,30:47='hasNameAsAttribute',<19>,1:30] )) ([@14,53:70='hasNameAsOperation',<19>,1:53] ))")
    print(ast.ExpressionString)
    print(ast.GetReferences(), "\n")
    ast = AST("([@6,22:28='implies',<59>,1:22] ([@2,14:15='<>',<30>,1:14] ([@0,0:12='specification',<19>,1:0] ) ([@4,17:20='NULL',<19>,1:17] )) ([@12,58:58='=',<13>,1:58] ([@8,30:54='ownedParameterPOINTERsize',<19>,1:30] ) ([@14,60:106='specificationSEPERATORownedParameterPOINTERsize',<19>,1:60] )))")
    print(ast.ExpressionString)
    print(ast.GetReferences(), "\n")
    ast = AST("([@0,0:33='librarySEPERATORloansPOINTERselect',<19>,1:0] ([@5,41:41='=',<13>,1:41] ([@3,36:39='book',<19>,1:36] ) ([@7,43:46='self',<19>,1:43] )))")
    print(ast.ExpressionString)
    print(ast.GetReferences(), "\n")
    ast = AST("([@0,0:19='loansPOINTERisUnique',<19>,1:0] ([@2,21:24='book',<19>,1:21] ))")
    print(ast.ExpressionString)
    print(ast.GetReferences(), "\n")

# Write to file
def WriteToFile():
    dao = DAO()
    dao.ChangeDB('FinalDB.db')
    constraints = dao.GetExpressions()
    counter = 1
    ExpressionsTxt = open("ValidExpressions.txt",'a',encoding="utf-8")
    for Exp in constraints:
        # ExpressionsTxt.write("@"+str(Exp[0])+"#"+str(Exp[1].replace(" ","").replace("\n","").replace("\t",""))+"\n")
        ExpressionsTxt.write("@" + str(Exp[0]) + "#" + str(Exp[1]) + "\n")


# Reading from File - ValidAST
def AddtoDBFromFile():
    file1 = open('ValidAST.txt', 'r')
    Lines = file1.readlines()
    dao = DAO()
    dao.ChangeDB('FinalDB.db')
    print(len(Lines))
    count = 0
    # Strips the newline character
    for line in Lines:
        SplitLine = line.split("$",maxsplit=2)
        ast = AST(SplitLine[1])
        print(ast.ExpressionString)
        dao.AddAST(SplitLine[0],SplitLine[1])
        dao.AddReferences(SplitLine[0],', '.join(ast.GetReferences()))

#
# CMD stuff
def CMD(constraints):
    counter = 0
    dao = DAO()
    dao.ChangeDB('FinalDB.db')
    for Exp in constraints:
        Command = "java -jar OCLtoAST.jar " + str(Exp[1].replace(" ",""))
        Output = subprocess.run(Command, capture_output=True, text=True)
        print(str(counter)+" - "+Output.stdout)
        counter += 1
        dao.AddAST(Exp[0], Output.stdout)

    Command = "java -jar OCLtoAST.jar " + "region->size()=0"
    Output = subprocess.run(Command, capture_output=True, text=True)
    print(Output.stdout)



#Model/Role Objects = (Name, ObjectID)
#ExpRef = (ConstraintID,ConstraintReferences,ModelID,ObjectID)
def CheckReferencesInConstraints():
    dao = DAO()
    dao.ChangeDB('FinalDB.db')
    dao.resetConstraintReferences()
    # dao.AddReferencedCol()
    Refs = dao.GetExpressionReferences()
    currModelID = -1
    currObjID = -1
    ModelObjects = None
    ObjectRoles = None
    counter = 0
    for ExpRef in Refs:
        if currModelID != int(ExpRef[2]):
            currModelID = int(ExpRef[2])
            ModelObjects = dao.GetModelObjects(currModelID)
        if currObjID != int(ExpRef[3]):
            currObjID = int(ExpRef[3])
            ObjectRoles = dao.GetObjectRoles(currObjID)
        for ObjRef in ExpRef[1].split(','):
            ObjRef = ObjRef.replace('->size','')
            if(ObjRef != ''):
                SplitObjRef = ObjRef.split('.')
                if len(SplitObjRef)>1:
                    for PartialRef in SplitObjRef:
                        for Role in ObjectRoles:
                            if PartialRef==Role[0]:
                                # print("found - "+str(PartialRef)+" = "+str(Role[0])+" = "+str(Role[1]))
                                dao.AddReferenced(Role[1])
                                try:
                                    if Role[1] == ExpRef[3]:
                                        dao.AddConstraintReference(currModelID,Role[1],ExpRef[0],1)
                                    else:
                                        dao.AddConstraintReference(currModelID, Role[1], ExpRef[0], 0)
                                    counter += 1
                                except:
                                    print(currModelID, Object[1], ExpRef[0])
                        for Object in ModelObjects:
                            if PartialRef==Object[0]:
                                # print("found - "+str(PartialRef)+" = "+str(Object[0])+" = "+str(Object[1]))
                                try:
                                    dao.AddReferenced(Object[1])
                                    if Object[1] == ExpRef[3]:
                                        dao.AddConstraintReference(currModelID, Object[1], ExpRef[0], 1)
                                    else:
                                        dao.AddConstraintReference(currModelID, Object[1], ExpRef[0], 0)
                                    counter += 1
                                except:
                                    print(currModelID, Object[1], ExpRef[0])
                else:
                    for Role in ObjectRoles:
                        if ObjRef == Role[0]:
                            # print("found - " + str(ObjRef) + " = " + str(Role[0]) + " = " + str(Role[1]))
                            dao.AddReferenced(Role[1])
                            try:
                                if Role[1] == ExpRef[3]:
                                    dao.AddConstraintReference(currModelID, Role[1], ExpRef[0], 1)
                                else:
                                    dao.AddConstraintReference(currModelID, Role[1], ExpRef[0], 0)

                                counter += 1
                            except:
                                print(currModelID, Object[1], ExpRef[0])
                    for Object in ModelObjects:
                        if ObjRef == Object[0]:
                            # print("found - " + str(ObjRef) + " = " + str(Object[0]) + " = " + str(Object[1]))
                            dao.AddReferenced(Object[1])
                            try:
                                if Object[1] == ExpRef[3]:
                                    dao.AddConstraintReference(currModelID, Object[1], ExpRef[0], 1)
                                else:
                                    dao.AddConstraintReference(currModelID, Object[1], ExpRef[0], 0)
                            except:
                                print(currModelID, Object[1], ExpRef[0])
                                counter += 1
    print(counter)

CheckReferencesInConstraints()
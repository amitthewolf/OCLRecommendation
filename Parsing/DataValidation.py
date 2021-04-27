from DAO import DAO

def getAllObjects():
    dao = DAO()
    dao.ChangeDB('LB_DB.db')
    ObjectConstraintsNums = dao.getObjectsConstNum()
    print("Objects with constraints: ",len(ObjectConstraintsNums))
    counter = 0
    ModelsWithProblems = {}
    for ObjConsNum in ObjectConstraintsNums:
        RealConstNum = dao.getRealConstNum(ObjConsNum[0])
        if RealConstNum[0][0] != ObjConsNum[2]:
            counter += 1
            if ObjConsNum[1] not in ModelsWithProblems:
                ModelsWithProblems[ObjConsNum[1]] = 1
            else:
                ModelsWithProblems[ObjConsNum[1]] = int(ModelsWithProblems[ObjConsNum[1]]) + 1
    print("Objects with constraints not in Constraints: ",counter)
    print(ModelsWithProblems)

def CheckConstraintsOrigins():
    dao = DAO()
    dao.ChangeDB('LB_DB.db')
    ObjectConstraintsNums = dao.getObjectsConstNum()
    ConstraintsOrigins = dao.GetConstraintOrigins()
    print("constraints: ",len(ConstraintsOrigins))
    counter = 0
    ConstraintswithnoOrigin = []
    for ConstraintOrigin in ConstraintsOrigins:
        Obj = dao.CheckIfObjectExists(ConstraintOrigin[2])
        if Obj!=1:
            counter += 1
            ConstraintswithnoOrigin.append(str(ConstraintOrigin[0])+","+str(ConstraintOrigin[2]))
    print("constraints without origins: "+str(counter))
    print(ConstraintswithnoOrigin)


CheckConstraintsOrigins()
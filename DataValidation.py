from DAO import DAO

def getAllObjects():
    dao = DAO()
    dao.ChangeDB('FinalDB.db')
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

getAllObjects()
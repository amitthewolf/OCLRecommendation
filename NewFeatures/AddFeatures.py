from DAO import DAO


def AddPointedAt():
    dao = DAO()
    dao.AddPointedAt()
    PointedAtObjs = dao.GetPointedAtRelations()
    CountingDict = {}
    for PointedAt in PointedAtObjs:
        if PointedAt[0] in CountingDict.keys():
            CountingDict[PointedAt[0]] = CountingDict[PointedAt[0]] + 1
        else:
            CountingDict[PointedAt[0]] = 1
    for key in CountingDict.keys():
        dao.SetPointedAt(CountingDict[key],key)
    dao.SetPointedAtNull()

def AddContianmentNum():
    dao = DAO()
    dao.AddContainmentNum()
    PointedAtObjs = dao.GetContainmentRelations()
    CountingDict = {}
    for PointedAt in PointedAtObjs:
        if PointedAt[0] in CountingDict.keys():
            CountingDict[PointedAt[0]] = CountingDict[PointedAt[0]] + 1
        else:
            CountingDict[PointedAt[0]] = 1
    for key in CountingDict.keys():
        dao.SetContainmentNum(CountingDict[key],key)
    dao.SetContainmentNumNull()


AddPointedAt()
AddContianmentNum()

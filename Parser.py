import xml.etree.ElementTree as ET
import os
from DAO import DAO
from re import search, IGNORECASE
from datetime import datetime

xsi = "{http://www.w3.org/2001/XMLSchema-instance}"
xmi = "{http://www.omg.org/XMI}"
LBpath = "C:/FinalProject/ModelDatabase/ocl-dataset-master/dataset/repos"
Amitpath = "C:/Uni/Final Project/Dataset/ocl-dataset-master/dataset/repos"
ohadPath = "C:/Users/ohadv/Desktop/ocl-dataset-master/dataset"


def GetName(Element):
    EleAtts = Element.attrib
    try:
        EleName = EleAtts.__getitem__("name")
        return EleName
    except:
        return "No Name"


def GetXSIType(Element):
    EleAtts = Element.attrib
    EleType = EleAtts.__getitem__(xsi + "type")
    return EleType


def GetXMIType(Element):
    EleAtts = Element.attrib
    EleType = EleAtts.__getitem__(xmi + "type")
    return EleType


def GetSource(Element):
    EleAtts = Element.attrib
    EleSource = EleAtts.__getitem__("source")
    return EleSource


def GetKey(Element):
    EleAtts = Element.attrib
    EleKey = EleAtts.__getitem__("key")
    return EleKey


def GetValue(Element):
    EleAtts = Element.attrib
    EleValue = EleAtts.__getitem__("value")
    return EleValue


def GeteType(Element):
    EleAtts = Element.attrib
    EleeType = EleAtts.__getitem__("eType")
    Split = EleeType.split("/")
    return Split[len(Split) - 1]


def GetType(Element):
    EcoreType = ""
    try:
        EcoreType = GetXSIType(Element)
    except:
        try:
            EcoreType = GetXMIType(Element)
        except:
            EcoreType = "None"
    return EcoreType


# os.chdir(LBpath)

# init DB
Dao = DAO()
Dao.resetRelations()
Dao.resetObjects()
Dao.resetConstraints()
Dao.resetModels()

# init counters
ObjectCounter = 0
FileCounter = 0
RelationCounter = 0
ModelCounter = 0
ConstraintsCounter = 0
Errors = 0
OCLFileCounter = 0
NoOCLFileCounter = 0
ModelsWithOCL = 0
ModelsWithoutOCL = 0
ObjectsinFileCounter = 0
OclInModelNum = 0

# init vars
ModelName = ""
MODELLLL = ""
LastRootName = ""
LastMODELLL = ""
OCLInModel = False
ObjectDic = {}

time = datetime.now()

for root, subdir, files in os.walk(Amitpath):
    for filename in files:
        if search(r'.*\.(ecore)$', filename, IGNORECASE):
            OCLFound = False
            try:
                FileCounter = FileCounter + 1
                Tree = ET.parse(root + "/" + filename)
                Root = Tree.getroot()
                MODELLLL = root
                if MODELLLL != LastMODELLL:
                    ModelCounter = ModelCounter + 1
                    ObjectDic.clear()
                    print(ModelCounter)
                    print(datetime.now() - time)
                    if OCLInModel:
                        ModelsWithOCL = ModelsWithOCL + 1
                        Dao.AddModel(ModelsWithOCL, MODELLLL, OclInModelNum, 0)
                        OclInModelNum = 0
                    else:
                        ModelsWithoutOCL = ModelsWithoutOCL + 1
                        Dao.RemoveModel(LastMODELLL)
                    LastMODELLL = MODELLLL
                    OCLInModel = False
                    ObjectsinFileCounter = 0
                for Class in Root.findall('eClassifiers'):
                    ClassName = GetName(Class)
                    ClassType = GetType(Class)
                    if ClassType == "ecore:EClass":
                        ObjectCounter = ObjectCounter + 1
                        name = ClassName
                        ID = ObjectCounter
                        ObjectDic[name] = ID
                for Class in Root.findall('eClassifiers'):
                    ClassName = GetName(Class)
                    ClassType = GetType(Class)
                    if ClassType == "ecore:EClass":
                        ObjectName = ClassName
                        ModelName = root
                        RelationNum = 0
                        AttNum = 0
                        ConstraintsCounter = 0
                        for Element in list(Class.iter()):
                            if Element.tag == "eStructuralFeatures":
                                EcoreType = GetType(Element)
                                if EcoreType == "ecore:EReference":
                                    Dao.AddRelation(root + "/" + filename, ModelName, Element, ObjectDic.get(ClassName),
                                                    ObjectDic.get(GeteType(Element)))
                                    RelationCounter = RelationCounter + 1
                                    RelationNum = RelationNum + 1
                                else:
                                    AttNum = AttNum + 1
                            if Element.tag == "eAnnotations":
                                EcoreSource = GetSource(Element)
                                if EcoreSource == "http://www.eclipse.org/emf/2002/Ecore/OCL/Pivot" or EcoreSource == "http://www.eclipse.org/emf/2002/Ecore/OCL":
                                    for SubElement in list(Element.iter()):
                                        if SubElement.tag == "details":
                                            ConstraintName = GetKey(SubElement)
                                            ConstraintExp = GetValue(SubElement)
                                            OCLFound = True
                                            OCLInModel = True
                                            Dao.AddConstraint(root + "/" + filename, ObjectName,
                                                              ObjectDic.get(ClassName), ConstraintName, ConstraintExp)
                                            ConstraintsCounter = ConstraintsCounter + 1
                                            OclInModelNum += 1
                                else:
                                    if EcoreSource == "http://www.eclipse.org/emf/2002/GenModel":
                                        try:
                                            for SubElement in list(Element.iter()):
                                                if SubElement.tag == "details":
                                                    ConstraintName = GetKey(SubElement)
                                                    ConstraintExp = GetValue(SubElement)
                                                    if ConstraintExp.__contains__("()"):
                                                        OCLFound = True
                                                        OCLInModel = True
                                                        Dao.AddConstraint(root + "/" + filename, ObjectName,
                                                                          ObjectDic.get(ClassName), ConstraintName,
                                                                          ConstraintExp)
                                                        ConstraintsCounter = ConstraintsCounter + 1
                                                        OclInModelNum += 1
                                        except:
                                            print("Annotation error")
                        ObjectsinFileCounter += 1
                        if RelationNum == 0:
                            Dao.AddObject(ObjectDic[ClassName], root + "/" + filename, ObjectName, ModelName,
                                          RelationNum, 0, AttNum, "", ConstraintsCounter)
                        else:
                            Dao.AddObject(ObjectDic[ClassName], root + "/" + filename, ObjectName, ModelName,
                                          RelationNum, RelationCounter, AttNum, "", ConstraintsCounter)
                if OCLFound:
                    OCLFileCounter = OCLFileCounter + 1
                else:
                    NoOCLFileCounter = NoOCLFileCounter + 1
            except Exception as e:
                print(e)
                Errors = Errors + 1

print("End - ", "")
print(datetime.now() - time)
print("Models:" + str(ModelCounter),
      "Models With Ocl: " + str(ModelsWithOCL) + ", Models Without OCL:" + str(ModelsWithoutOCL))
print("Files:" + str(FileCounter), "Errors: " + str(Errors) + ", Files With OCL:" + str(OCLFileCounter))
Dao.conn.commit()
Dao.conn.close()

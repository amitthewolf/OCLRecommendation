import xml.etree.ElementTree as ET
import os
from DAO import DAO
from re import search, IGNORECASE

xsi = "{http://www.w3.org/2001/XMLSchema-instance}"
xmi = "{http://www.omg.org/XMI}"
path = "C:/Users/amitt/Desktop/ThreeEyes/ocl-dataset-master/dataset/repos"


def  GetName(Element):
    EleAtts = Element.attrib
    try:
        EleName = EleAtts.__getitem__("name")
        return EleName
    except:
        return "No Name"

def  GetXSIType(Element):
    EleAtts = Element.attrib
    EleType = EleAtts.__getitem__(xsi+"type")
    return EleType

def  GetXMIType(Element):
    EleAtts = Element.attrib
    EleType = EleAtts.__getitem__(xmi+"type")
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
    return Split[len(Split)-1]


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


Dao = DAO()
Dao.resetRelations()
Dao.resetObjects()
Dao.resetConstraints()
os.chdir(path)
ObjectCounter = 0
FileCounter = 0
RelationCounter = 0
ConstraintsCounter = 0;
Errors = 0
OCLFileCounter = 0
ObjectDic = {}

for root, subdir, files in os.walk(path):
    for filename in files:
        if search(r'.*\.(ecore)$', filename, IGNORECASE):
            OCLFound = False
            try:
                FileCounter = FileCounter + 1
                Tree = ET.parse(root+"/"+filename)
                Root = Tree.getroot()
                RootName = GetName(Root)
                ObjectDic.clear()
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
                        # print("Another Object : " + str(FileCounter + 1))
                        ObjectName = ClassName + "-" + RootName
                        # print(ObjectName)
                        RelationNum = 0
                        AttNum = 0
                        ConstraintsCounter = 0
                        for Element in list(Class.iter()):
                            if Element.tag == "eStructuralFeatures":
                                EcoreType = GetType(Element)
                                if EcoreType == "ecore:EReference":
                                    Dao.AddRelation(root+"/"+filename,Element, ObjectDic.get(ClassName),ObjectDic.get(GeteType(Element)))
                                    RelationCounter = RelationCounter + 1
                                    RelationNum = RelationNum + 1
                                else:
                                    AttNum = AttNum + 1
                            if Element.tag == "eAnnotations":
                                EcoreSource = GetSource(Element)
                                if EcoreSource == "http://www.eclipse.org/emf/2002/Ecore/OCL/Pivot" or EcoreSource == "http://www.eclipse.org/emf/2002/Ecore/OCL":
                                    for SubElement in list(Element.iter()):
                                        if(SubElement.tag == "details"):
                                            ConstraintName = GetKey(SubElement)
                                            ConstraintExp = GetValue(SubElement)
                                            OCLFound = True
                                            Dao.AddConstraint(root+"/"+filename,ObjectName,ObjectDic.get(ClassName),ConstraintName,ConstraintExp)
                                            ConstraintsCounter = ConstraintsCounter + 1
                                else:
                                    if EcoreSource=="http://www.eclipse.org/emf/2002/GenModel":
                                        try:
                                            for SubElement in list(Element.iter()):
                                                if (SubElement.tag == "details"):
                                                    ConstraintName = GetKey(SubElement)
                                                    ConstraintExp = GetValue(SubElement)
                                                    if(ConstraintExp.__contains__("()")):
                                                        OCLFound = True
                                                        Dao.AddConstraint(root + "/" + filename, ObjectName, ObjectDic.get(ClassName), ConstraintName, ConstraintExp)
                                                        ConstraintsCounter = ConstraintsCounter + 1
                                        except:
                                            print("Annotation error")

                        if RelationNum == 0:
                            Dao.AddObject(root + "/" + filename, ObjectName, RelationNum, 0, AttNum, "", ConstraintsCounter)
                        else:
                            Dao.AddObject(root+"/"+filename,ObjectName, RelationNum, RelationCounter, AttNum, "", ConstraintsCounter)
                if(OCLFound):
                    OCLFileCounter = OCLFileCounter + 1
                print("Another")
            except:
                Errors = Errors + 1


print("Files:"+str(FileCounter),"Errors: "+str(Errors)+", Files With OCL:"+str(OCLFileCounter))
Dao.conn.commit()
Dao.conn.close()

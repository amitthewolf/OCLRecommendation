
import xml.etree.ElementTree as ET
import sqlite3
import os
from re import search, IGNORECASE

xsi = "{http://www.w3.org/2001/XMLSchema-instance}"
xmi = "{http://www.omg.org/XMI}"

#Functions

def  AddRelation(FileLocation,Relation,ParentID,ReferenceID):
    RelationAtt = Relation.attrib
    upperBound = ""
    lowerBound = ""
    containment = ""
    try:
        upperBound = str(RelationAtt.__getitem__("upperBound"))

    except:
        upperBound = ""
    try:
        lowerBound = RelationAtt.__getitem__("lowerBound")

    except:
        lowerBound = ""
    try:
        containment = RelationAtt.__getitem__("containment")

    except:
        containment = ""
    c.execute(" INSERT INTO Relations (FileLocation,ObjectID1,ObjectID2,LowerBound,UpperBound, Containment) VALUES (?,?,?,?,?,?)",
              (FileLocation,ParentID,ReferenceID,lowerBound,upperBound,containment))

def  AddObject(FileLocation,ObjectName,RelationNum,LastRelationID,AttributeNum,SemanticWords,ConstraintsNum):
    c.execute(" INSERT INTO Objects (FileLocation,ObjectName,RelationNum,LastRelationID,AttributeNum,SemanticWords,ConstraintsNum) VALUES (?,?,?,?,?,?,?)",
              (FileLocation,ObjectName,RelationNum,LastRelationID,AttributeNum,SemanticWords,ConstraintsNum))

def  AddConstraint(FileLocation,ObjectName,ObjectID,ConstraintName,Expression):
    c.execute(" INSERT INTO Constraints (FileLocation,ObjectName,ObjectID,ConstraintName,Expression) VALUES (?,?,?,?,?)",
              (FileLocation,ObjectName,ObjectID,ConstraintName,Expression))

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

def  resetRelations():
    c.execute("drop table Relations")
    c.execute(""" CREATE TABLE Relations (
                  RelationID integer PRIMARY KEY AUTOINCREMENT,
                   FileLocation text,
                  ObjectID1 integer,
                  ObjectID2 integer,
                  LowerBound integer,
                  UpperBound integer,
                  Containment integer
                  )""")

def resetObjects():
    c.execute("drop table Objects")
    c.execute(""" CREATE TABLE Objects (
                    ObjectID integer primary key AUTOINCREMENT,
                     FileLocation text,
                  ObjectName text,
                  RelationNum integer,
                  LastRelationID integer,
                  AttributeNum integer,
                  SemanticWords String,
                  ConstraintsNum integer)""")

def resetConstraints():
    c.execute("drop table Constraints")
    c.execute(""" CREATE TABLE Constraints (
                  ConstraintID integer primary key AUTOINCREMENT,
                  FileLocation text,
                  ObjectName text,
                  ObjectID integer,
                  ConstraintName text,
                  Expression text )""")


#Reset Database and Initialization
conn = sqlite3.connect('ThreeEyesDB.db')
c = conn.cursor()
resetRelations()
resetObjects()
resetConstraints()
os.chdir("C:/Uni/Final Project/Dataset/ocl-dataset-master/dataset/repos")
ObjectCounter = 0
FileCounter = 0
RelationCounter = 0
ConstraintsCounter = 0;
Errors = 0
OCLFileCounter = 0
ObjectDic = {}

#Parsing
for root, subdir, files in os.walk("C:/Uni/Final Project/Dataset/ocl-dataset-master/dataset/repos"):
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
                                    AddRelation(root+"/"+filename,Element, ObjectDic.get(ClassName),ObjectDic.get(GeteType(Element)))
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
                                            AddConstraint(root+"/"+filename,ObjectName,ObjectDic.get(ClassName),ConstraintName,ConstraintExp)
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
                                                        AddConstraint(root + "/" + filename, ObjectName, ObjectDic.get(ClassName), ConstraintName, ConstraintExp)
                                                        ConstraintsCounter = ConstraintsCounter + 1
                                        except:
                                            print("Annotation error")

                        if RelationNum == 0:
                            AddObject(root + "/" + filename, ObjectName, RelationNum, 0, AttNum, "", ConstraintsCounter)
                        else:
                            AddObject(root+"/"+filename,ObjectName, RelationNum, RelationCounter, AttNum, "", ConstraintsCounter)
                if(OCLFound):
                    OCLFileCounter = OCLFileCounter + 1
                print("Another")
            except:
                Errors = Errors + 1









print("Files:"+str(FileCounter),"Errors: "+str(Errors)+", Files With OCL:"+str(OCLFileCounter))
conn.commit()
conn.close()

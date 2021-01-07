import xml.etree.ElementTree as ET
import os
from DAO import DAO
from re import search, IGNORECASE
from datetime import datetime


class Parser:

    def __init__(self):

        self.ohadPath = "C:/Users/ohadv/Desktop/FinalProject/ocl-dataset-master/dataset"
        self.Amitpath = "C:/Uni/Final Project/Dataset/ocl-dataset-master/dataset/repos"
        self.LapTopAmit = "C:/Users/amitt/Desktop/ThreeEyes/ocl-dataset-master/dataset/repos"
        self.LB_Path = "C:/FinalProject/ModelDatabase/ocl-dataset-master/dataset/repos"
        self.ohadLaptop = "D:/ocl-dataset-master/dataset/repos"

        self.xsi = "{http://www.w3.org/2001/XMLSchema-instance}"
        self.xmi = "{http://www.omg.org/XMI}"

        self.dao = DAO()
        self.ObjectDic = {}

        self.ConstraintsCounter = 0
        self.OclInModelNum = 0
        self.ObjectCounter = 0
        self.FileCounter = 0
        self.RelationCounter = 0
        self.ModelCounter = 0
        self.Errors = 0
        self.OCLFileCounter = 0
        self.NoOCLFileCounter = 0
        self.ModelsWithOCL = 1
        self.ModelsWithoutOCL = 0
        self.ObjectsinFileCounter = 0
        self.ObjectsinModel = 0

        self.constraints_in_obj = 0

        self.RelationNum = 0
        self.AttNum = 0
        self.super = ""
        self.abstract = 0
        self.properties = ""

        self.model_hash_value = 0
        self.model_hashes = list()

        # indicates whether we want to keep duplicated models in our data set
        self.keep_duplicates = False

    def GetName(self, Element):
        EleAtts = Element.attrib
        try:
            EleName = EleAtts.__getitem__("name")
            return EleName
        except:
            return "No Name"

    def GetXSIType(self, Element):
        EleAtts = Element.attrib
        EleType = EleAtts.__getitem__(self.xsi + "type")
        return EleType

    def GetXMIType(self, Element):
        EleAtts = Element.attrib
        EleType = EleAtts.__getitem__(self.xmi + "type")
        return EleType

    def GetSource(self, Element):
        EleAtts = Element.attrib
        EleSource = EleAtts.__getitem__("source")
        return EleSource

    def GetKey(self, Element):
        EleAtts = Element.attrib
        EleKey = EleAtts.__getitem__("key")
        return EleKey

    def GetValue(self, Element):
        EleAtts = Element.attrib
        EleValue = EleAtts.__getitem__("value")
        return EleValue

    def GeteType(self, Element):
        EleAtts = Element.attrib
        EleeType = EleAtts.__getitem__("eType")
        Split = EleeType.split("/")
        return Split[len(Split) - 1]

    def GetType(self, Element):
        EcoreType = ""
        try:
            EcoreType = self.GetXSIType(Element)
        except:
            try:
                EcoreType = self.GetXMIType(Element)
            except:
                EcoreType = "None"
        return EcoreType

    def handleAnnotation(self, Element, ObjectName, ClassName):
        if Element.tag == "eAnnotations":
            flag = False
            EcoreSource = self.GetSource(Element)
            if EcoreSource == "http://www.eclipse.org/emf/2002/Ecore/OCL/Pivot" or EcoreSource == "http://www.eclipse.org/emf/2002/Ecore/OCL":
                for SubElement in list(Element.iter()):
                    if SubElement.tag == "details":
                        ConstraintName = self.GetKey(SubElement)
                        ConstraintExp = self.GetValue(SubElement)
                        flag = True
                        self.constraints_in_obj += 1
                        self.OclInModelNum += 1
                        self.ConstraintsCounter += 1
                        self.dao.AddConstraint(self.ConstraintsCounter, self.ModelsWithOCL, ObjectName,
                                               self.ObjectDic.get(ClassName), "1", ConstraintName, ConstraintExp)

            else:
                if EcoreSource == "http://www.eclipse.org/emf/2002/GenModel":
                    try:
                        for SubElement in list(Element.iter()):
                            if SubElement.tag == "details":
                                ConstraintName = self.GetKey(SubElement)
                                ConstraintExp = self.GetValue(SubElement)
                                if ConstraintExp.__contains__("()"):
                                    flag = True
                                    self.dao.AddConstraint(self.ConstraintsCounter, self.ModelsWithOCL, ObjectName,
                                                           self.ObjectDic.get(ClassName), "1", ConstraintName,
                                                           ConstraintExp)
                                    self.constraints_in_obj += 1
                                    self.ConstraintsCounter += 1
                                    self.OclInModelNum += 1
                    except:
                        print("Annotation error")
            return flag

    def createObjectDictionary(self, Class):
        ClassName = self.GetName(Class)
        ClassType = self.GetType(Class)
        if ClassType == "ecore:EClass":
            self.ObjectCounter += 1
            name = ClassName
            ID = self.ObjectCounter
            self.ObjectDic[name] = ID

    def handleRelation(self, Element, ClassName, ModelName):
        if Element.tag == "eStructuralFeatures":
            EcoreType = self.GetType(Element)
            if EcoreType == "ecore:EReference":
                self.dao.AddRelation(self.ModelsWithOCL, ModelName, Element, self.ObjectDic.get(ClassName),
                                     self.ObjectDic.get(self.GeteType(Element)))
                self.RelationCounter += 1
                self.RelationNum += 1
            elif EcoreType == "ecore:EAttribute":
                self.properties += Element.attrib.__getitem__("name") + ","
                self.AttNum += 1

    def handle_super(self, Element):
        if Element.tag == "eClassifiers":
            try:
                self.super = Element.attrib.__getitem__("eSuperTypes").split('/')[-1]
                self.super = self.ObjectDic[self.super]
                self.abstract = Element.attrib.__getitem__("abstract")
                if self.abstract != 0:
                    self.abstract = 1
            except:
                pass

    def parse(self):

        # init DB
        self.dao.resetRelations()
        self.dao.resetObjects()
        self.dao.resetConstraints()
        self.dao.resetModels()

        # init vars
        ModelName = ""
        MODELLLL = ""
        LastRootName = ""
        LastMODELLL = ""
        OCLInModel = False

        time = datetime.now()
        for root, subdir, files in os.walk(self.LB_Path):
            for filename in files:
                if search(r'.*\.(ecore)$', filename, IGNORECASE):
                    OCLFound = False
                    try:
                        self.FileCounter += 1
                        Tree = ET.parse(root + "/" + filename)
                        Root = Tree.getroot()
                        MODELLLL = root
                        if MODELLLL != LastMODELLL and LastMODELLL != "":
                            self.ModelCounter += 1
                            self.model_hash_value = hash(frozenset(self.ObjectDic.keys()))
                            self.ObjectDic.clear()
                            print(self.ModelCounter)
                            # print(datetime.now() - time)

                            # Dealing with non-ocl models(add to db/remove)
                            if OCLInModel:
                                if (self.model_hash_value not in self.model_hashes) or self.keep_duplicates:
                                    self.model_hashes.append(self.model_hash_value)
                                    self.dao.AddModel(self.ModelsWithOCL, MODELLLL, self.OclInModelNum, self.ObjectsinModel,
                                                  0, self.model_hash_value)
                                    self.ModelsWithOCL += 1
                                else:
                                    self.dao.RemoveConstraints(self.ModelsWithOCL)
                                self.OclInModelNum = 0
                                self.ObjectsinModel = 0
                            else:
                                self.ModelsWithoutOCL += 1
                                self.dao.RemoveModel(self.ModelsWithOCL)
                                self.ObjectsinModel = 0
                            OCLInModel = False
                            self.ObjectsinFileCounter = 0
                        LastMODELLL = MODELLLL
                        # First iteration on all model objects for creating object dictionary.
                        for Class in Root.findall('eClassifiers'):
                            self.createObjectDictionary(Class)

                        # Second iteration on all model objects
                        for Class in Root.findall('eClassifiers'):
                            ClassName = self.GetName(Class)
                            ClassType = self.GetType(Class)
                            if ClassType == "ecore:EClass":
                                ObjectName = ClassName
                                ModelName = root
                                self.RelationNum = 0
                                self.AttNum = 0
                                self.properties = ""
                                self.super = ""
                                self.abstract = 0
                                self.constraints_in_obj = 0
                                for Element in list(Class.iter()):
                                    self.handle_super(Element)
                                    self.handleRelation(Element, ClassName, ModelName)
                                    if self.handleAnnotation(Element, ObjectName, ClassName):
                                        OCLFound = True
                                        OCLInModel = True
                                self.ObjectsinFileCounter += 1
                                self.ObjectsinModel += 1
                                if self.RelationNum == 0:
                                    self.dao.AddObject(self.ObjectDic[ClassName], self.ModelsWithOCL, ObjectName,
                                                       ModelName + "/" + filename, self.RelationNum, 0, self.AttNum, "",
                                                       self.constraints_in_obj, self.properties, self.super, self.abstract)
                                else:
                                    self.dao.AddObject(self.ObjectDic[ClassName], self.ModelsWithOCL, ObjectName,
                                                       ModelName + "/" + filename, self.RelationNum,
                                                       self.RelationCounter, self.AttNum, "", self.constraints_in_obj,
                                                       self.properties, self.super, self.abstract)
                        if OCLFound:
                            self.OCLFileCounter += 1
                        else:
                            self.NoOCLFileCounter += 1
                    except Exception as e:
                        print(e)
                        self.Errors += 1

        # print(datetime.now() - time)
        print("Models:" + str(self.ModelCounter))
        print("Models With Ocl: " + str(self.ModelsWithOCL))
        print("Models Without OCL:" + str(self.ModelsWithoutOCL))
        print("------------")
        print("Files:" + str(self.FileCounter))
        print("Errors: " + str(self.Errors))
        print("Files With OCL:" + str(self.OCLFileCounter))
        self.dao.conn.commit()
        self.dao.conn.close()

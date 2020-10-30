import sqlite3
import pandas as pd


class DAO:

    def __init__(self):
        self.conn = sqlite3.connect('ThreeEyesDB.db')
        self.c = self.conn.cursor()

    def resetRelations(self):
        self.c.execute("drop table Relations")
        self.c.execute(""" CREATE TABLE Relations (
                      RelationID integer PRIMARY KEY AUTOINCREMENT,
                      FileLocation text,
                      ModelName text,
                      ObjectID1 integer,
                      ObjectID2 integer,
                      LowerBound integer,
                      UpperBound integer,
                      Containment integer
                      )""")

    def resetObjects(self):
        self.c.execute("drop table Objects")
        self.c.execute(""" CREATE TABLE Objects (
                        ObjectID integer primary key AUTOINCREMENT,
                         FileLocation text,
                      ObjectName text,
                      ModelName text,
                      RelationNum integer,
                      LastRelationID integer,
                      AttributeNum integer,
                      SemanticWords String,
                      ConstraintsNum integer)""")

    def resetConstraints(self):
        self.c.execute("drop table Constraints")
        self.c.execute(""" CREATE TABLE Constraints (
                      ConstraintID integer primary key,
                      FileLocation text,
                      ObjectName text,
                      ObjectID integer,
                      ConstraintName text,
                      Expression text )""")

    def AddRelation(self, FileLocation, ModelName, Relation, ParentID, ReferenceID):
        RelationAtt = Relation.attrib
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
        self.c.execute(
            "INSERT INTO Relations (FileLocation,ModelName,ObjectID1,ObjectID2,LowerBound,UpperBound, Containment) "
            "VALUES (?,?,?,?,?,?,?)",
            (FileLocation, ModelName, ParentID, ReferenceID, lowerBound, upperBound, containment))

    def AddObject(self, ObjectID, FileLocation, ObjectName, ModelName, RelationNum, LastRelationID, AttributeNum,
                  SemanticWords, ConstraintsNum):
        self.c.execute(
            "INSERT INTO Objects (ObjectID, FileLocation,ObjectName,ModelName,RelationNum,LastRelationID,"
            "AttributeNum,SemanticWords,ConstraintsNum) VALUES (?,?,?,?,?,?,?,?,?)",
            (ObjectID, FileLocation, ObjectName, ModelName, RelationNum, LastRelationID, AttributeNum, SemanticWords,
             ConstraintsNum))

    def AddConstraint(self, FileLocation, ObjectName, ObjectID, ConstraintName, Expression):
        self.c.execute(
            " INSERT INTO Constraints (FileLocation,ObjectName,ObjectID,ConstraintName,Expression) VALUES (?,?,?,?,?)",
            (FileLocation, ObjectName, ObjectID, ConstraintName, Expression))

    def RemoveModel(self, ModelName):
        self.c.execute(""" DELETE FROM Objects WHERE ModelName=?""", (ModelName,))
        self.c.execute(""" DELETE FROM Relations WHERE ModelName=?""", (ModelName,))

    def AddModel(self, ModelID, ModelName, ConstraintsNum, NormConstraints):
        self.c.execute(
            " INSERT INTO Models (ModelID, ModelName, ConstraintsNum, NormConstraints) VALUES (?,?,?,?)",
            (ModelID, ModelName, ConstraintsNum, NormConstraints))

    def resetModels(self):
        self.c.execute("drop table Models")
        self.c.execute(""" CREATE TABLE Models (
                         ModelID integer primary key,
                         ModelName text,
                         ConstraintsNum integer,
                         NormConstraints float)""")

    def getLargestModel(self):
        self.c.execute("Select MAX(ConstraintsNum) from Models")
        self.conn.commit()
        result = self.c.fetchall()
        return result[0][0]

    # def addColumnToModels(self):
    #     largetModel = self.getLargestModel()
    #     df = pd.read_sqlself.c.execute("Select * from Models")
    #     self.conn.commit()
    #     result = self.c.fetchall()
    #     print(result)

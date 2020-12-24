import sqlite3
import pandas as pd


class DAO:

    def __init__(self):
        self.conn = sqlite3.connect('ThreeEyesDB.db')
        self.c = self.conn.cursor()

    def ChangeDB(self, newDB):
        self.conn = sqlite3.connect(newDB)
        self.c = self.conn.cursor()

    def resetRelations(self):
        self.c.execute("drop table  if exists Relations")
        self.c.execute(""" CREATE TABLE Relations (
                      RelationID integer PRIMARY KEY AUTOINCREMENT,
                      ModelID integer,
                      ModelName text,
                      ObjectID1 integer,
                      ObjectID2 integer,
                      LowerBound integer,
                      UpperBound integer,
                      Containment integer
                      )""")

    def resetObjects(self):
        self.c.execute("drop table if exists Objects")
        self.c.execute(""" CREATE TABLE Objects (
                        ObjectID integer primary key AUTOINCREMENT,
                      ModelID integer,
                      ObjectName text,
                      ModelName text,
                      RelationNum integer,
                      LastRelationID integer,
                      AttributeNum integer,
                      SemanticWords String,
                      ConstraintsNum integer,
                      properties_names)""")

    def resetConstraints(self):
        self.c.execute("drop table if exists Constraints")
        self.c.execute(""" CREATE TABLE Constraints (
                      ConstraintID integer,
                      ModelID integer,
                      ObjectName text,
                      isContext bit,
                      ObjectID integer,
                      ConstraintName text,
                      Expression text,
                      OperationsNum integer,
                      ObjectsNum integer,
                      AST text,
                      primary key (ConstraintID, ModelID ,ObjectID))""")

        # def resetObj(self):
        #     self.c.execute("drop table if exists Constraints")
        #     self.c.execute(""" CREATE TABLE Constraints (
        #                      ConstraintID integer,
        #                      ModelID integer,
        #                      ObjectName text,
        #                      isContext bit,
        #                      ObjectID integer,
        #                      ConstraintName text,
        #                      Expression text,
        #                      AST text,
        #                      primary key (ConstraintID, ModelID ,ObjectID))""")

    def resetModels(self):
        self.c.execute("drop table if exists Models")
        self.c.execute(""" CREATE TABLE Models (
                         ModelID integer primary key,
                         ModelName text,
                         ConstraintsNum integer,
                         ObjectsNum integer,
                         NormConstraints float)""")

    def AddRelation(self, ModelID, ModelName, Relation, ParentID, ReferenceID):
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
            "INSERT INTO Relations (ModelID,ModelName,ObjectID1,ObjectID2,LowerBound,UpperBound, Containment) "
            "VALUES (?,?,?,?,?,?,?)",
            (ModelID, ModelName, ParentID, ReferenceID, lowerBound, upperBound, containment))

    def AddObject(self, ObjectID, ModelID, ObjectName, ModelName, RelationNum, LastRelationID, AttributeNum,
                  SemanticWords, ConstraintsNum, properties_names):
        self.c.execute(
            "INSERT INTO Objects (ObjectID, ModelID,ObjectName,ModelName,RelationNum,LastRelationID,"
            "AttributeNum,SemanticWords,ConstraintsNum, properties_names) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (ObjectID, ModelID, ObjectName, ModelName, RelationNum, LastRelationID, AttributeNum, SemanticWords,
             ConstraintsNum, properties_names))

    def AddConstraint(self, ConstraintID, ModelID, ObjectName, ObjectID, isContext, ConstraintName, Expression):
        self.c.execute(
            " INSERT INTO Constraints (ConstraintID, ModelID,ObjectName,ObjectID,isContext,ConstraintName,Expression) VALUES (?,?,?,?,?, ?, ?)",
            (ConstraintID, ModelID, ObjectName, ObjectID, isContext, ConstraintName, Expression))

    def GetExpressions(self):
        self.c.execute("SELECT ConstraintID,Expression from Constraints")
        self.conn.commit()
        result = self.c.fetchall()
        return result

    def Get_Expressions_For_Validation(self):
        self.c.execute("select Constraints.ConstraintID, Constraints.ConstraintName, Constraints.Expression,"
                       "Objects.ModelName, Objects.ObjectName from Constraints inner join "
                       "Objects on Constraints.ObjectID = Objects.ObjectID")
        self.conn.commit()
        result = self.c.fetchall()
        return result

    def AddAST(self, ConstraintID, AST):
        self.c.execute(
            " UPDATE Constraints SET AST = ? WHERE ConstraintID = ?", (AST, ConstraintID))
        self.conn.commit()

    def AddASTCol(self):
        self.c.execute(" ALTER TABLE Constraints ADD AST text")

    def RemoveModel(self, ModelID):
        self.c.execute(""" DELETE FROM Objects WHERE ModelID=?""", (ModelID,))
        self.c.execute(""" DELETE FROM Relations WHERE ModelID=?""", (ModelID,))

    def AddModel(self, ModelID, ModelName, ConstraintsNum, ObjectsNum, NormConstraints):
        self.c.execute(
            " INSERT INTO Models (ModelID, ModelName, ConstraintsNum,ObjectsNum, NormConstraints) VALUES (?,?,?,?,?)",
            (ModelID, ModelName, ConstraintsNum, ObjectsNum, NormConstraints))

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

    def delete_invalid_constraints(self, constraint_id):
        try:
            self.c.execute("SELECT ObjectID From Constraints WHERE ConstraintID=?", (constraint_id,))
            self.conn.commit()
            objectID = self.c.fetchall()[0][0]
        except:
            return
        self.c.execute(""" Select ConstraintsNum From Objects WHERE ObjectID=?""", (objectID,))
        self.conn.commit()
        ConstraintsNum = self.c.fetchall()[0][0]

        self.c.execute(""" UPDATE Objects SET ConstraintsNum=? Where ObjectID=?""",
                       ((ConstraintsNum - 1), objectID,))
        self.conn.commit()
        self.c.execute(""" DELETE FROM Constraints WHERE ConstraintID=?""", (constraint_id,))
        self.conn.commit()



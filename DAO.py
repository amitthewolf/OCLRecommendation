import sqlite3
import pandas as pd


class DAO:

    def __init__(self):
        self.conn = sqlite3.connect('Pipeline Database.db')
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
                      Role text,
                      LowerBound integer,
                      UpperBound integer,
                      Containment integer
                      )""")

    def resetConstraintReferences(self):
        self.c.execute("drop table  if exists ConstraintReferences")
        self.c.execute(""" CREATE TABLE ConstraintReferences (
                      ObjectID integer,
                      ModelID integer,
                      ConstraintID integer,
                      IsContext BIT,
                      primary key (ConstraintID, ModelID ,ObjectID))""")

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
                      properties_names,
                      inheriting_from,
                      is_abstract)""")

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
                      ConstraintReferences text,
                      Operators text,
                      primary key (ConstraintID, ModelID ,ObjectID))""")


    def rewriteObjectTable(self,df):
        df.to_sql('Objects', self.conn, if_exists='replace')

    def resetModels(self):
        self.c.execute("drop table if exists Models")
        self.c.execute(""" CREATE TABLE Models (
                         ModelID integer primary key,
                         ModelName text,
                         ConstraintsNum integer,
                         ObjectsNum integer,
                         NormConstraints float,
                         hashValue float)""")

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
        try:
            Role = RelationAtt.__getitem__("name")
        except:
            Role = ""
        self.c.execute(
            "INSERT INTO Relations (ModelID,ModelName,ObjectID1,ObjectID2,Role,LowerBound,UpperBound, Containment) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (ModelID, ModelName, ParentID, ReferenceID, Role, lowerBound, upperBound, containment))

    def AddObject(self, ObjectID, ModelID, ObjectName, ModelName, RelationNum, LastRelationID, AttributeNum,
                  SemanticWords, ConstraintsNum, properties_names, inheriting_from, is_abstract):
        self.c.execute(
            "INSERT INTO Objects (ObjectID, ModelID,ObjectName,ModelName,RelationNum,LastRelationID,"
            "AttributeNum,SemanticWords,ConstraintsNum, properties_names, inheriting_from, is_abstract) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (ObjectID, ModelID, ObjectName, ModelName, RelationNum, LastRelationID, AttributeNum, SemanticWords,
             ConstraintsNum, properties_names, inheriting_from, is_abstract))

    def AddConstraint(self, ConstraintID, ModelID, ObjectName, ObjectID, isContext, ConstraintName, Expression):
        self.c.execute(
            " INSERT INTO Constraints (ConstraintID, ModelID,ObjectName,ObjectID,isContext,ConstraintName,Expression) VALUES (?,?,?,?,?, ?, ?)",
            (ConstraintID, ModelID, ObjectName, ObjectID, isContext, ConstraintName, Expression))

    def GetExpressions(self):
        self.c.execute("SELECT ConstraintID,Expression from Constraints")
        self.conn.commit()
        result = self.c.fetchall()
        return result

    def getObjectsConstNum(self):
        self.c.execute("SELECT ObjectID,ModelID,ConstraintsNum FROM Objects WHERE ConstraintsNum > 0")
        self.conn.commit()
        result = self.c.fetchall()
        return result

    def getRealConstNum(self,ObjectID):
        self.c.execute("SELECT COUNT(ConstraintID) From Constraints WHERE ObjectID=?", (ObjectID,))
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
        self.conn.commit()


    def RemoveConstraints(self, ModelID):
        self.c.execute(""" Delete from constraints where ModelID=?""", (ModelID,))
        self.conn.commit()

    def AddModel(self, ModelID, ModelName, ConstraintsNum, ObjectsNum, NormConstraints, hashValue):
        self.c.execute(
            " INSERT INTO Models (ModelID, ModelName, ConstraintsNum,ObjectsNum, NormConstraints, hashValue) VALUES (?,?,?,?,?,?)",
            (ModelID, ModelName, ConstraintsNum, ObjectsNum, NormConstraints, hashValue))

    def getLargestModel(self):
        self.c.execute("Select MAX(ConstraintsNum) from Models")
        self.conn.commit()
        result = self.c.fetchall()
        return result[0][0]

    def getObjects(self):
        df = pd.read_sql("SELECT * FROM Objects", self.conn)
        return df

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

    def getObjectNames(self):
        self.c.execute("select ModelID, ObjectName from Objects")
        self.conn.commit()
        result = self.c.fetchall()
        return result

    def AddConstraintReference(self, ModelID, ObjectID, ConstraintID, isContext):
        self.c.execute(
            " INSERT INTO ConstraintReferences ( ModelID, ObjectID, ConstraintID, IsContext) VALUES (?,?,?,?)",
            (ModelID, ObjectID, ConstraintID, isContext))

    def AddConstraintOperatorsRow(self, ConstraintID):
        self.c.execute(
            " INSERT INTO ConstraintOperators ( ConstraintID) VALUES (?)",
            (ConstraintID,))

    def UpdateConstraintOpsCount(self,ConstraintID,Operator,OperatorCount):
        self.c.execute(""" UPDATE ConstraintOperators SET {}=? Where ConstraintID=?""".format(Operator),
                       (OperatorCount,ConstraintID))
        self.conn.commit()


    def GetExpressions(self):
        self.c.execute("SELECT ConstraintID,Expression from Constraints")
        self.conn.commit()
        result = self.c.fetchall()
        return result

    def GetConstraintOperatorsTable(self):
        self.c.execute("SELECT * from ConstraintOperators")
        self.conn.commit()
        result = self.c.fetchall()
        return result

    def GetExpressionReferences(self):
        self.c.execute("SELECT ConstraintID,ConstraintReferences,ModelID,ObjectID from Constraints")
        self.conn.commit()
        result = self.c.fetchall()
        return result

    def GetConstraintOperators(self):
        self.c.execute("SELECT ConstraintID,Operators from Constraints")
        self.conn.commit()
        result = self.c.fetchall()
        return result

    def UpdateConstraintOps(self,ConstraintID,Logical,OCL):
        self.c.execute(""" UPDATE Constraints SET LogicalOperators=?, OCLOperators=? Where ConstraintID = ? """,
                       (Logical,OCL,ConstraintID))
        self.conn.commit()

    def GetObjectRoles(self,ObjectID):
        self.c.execute("SELECT Role,ObjectID2 from Relations WHERE ObjectID1 = ?",(ObjectID,))
        self.conn.commit()
        result = self.c.fetchall()
        return result

    def GetModelObjects(self,ModelID):
        self.c.execute("SELECT ObjectName,ObjectID from Objects WHERE ModelID = ?",(ModelID,))
        self.conn.commit()
        result = self.c.fetchall()
        return result


    def AddAST(self, ConstraintID, AST):
        self.c.execute(
            " UPDATE Constraints SET AST = ? WHERE ConstraintID = ?",(AST,ConstraintID))
        self.conn.commit()

    def AddReferences(self, ConstraintID, References):
        self.c.execute(
            " UPDATE Constraints SET ConstraintReferences = ? WHERE ConstraintID = ?", (References, ConstraintID))
        self.conn.commit()

    def AddOperators(self, ConstraintID, Operators):
        self.c.execute(
            " UPDATE Constraints SET Operators = ? WHERE ConstraintID = ?", (Operators, ConstraintID))
        self.conn.commit()

    def AddOperators(self, ConstraintID, Operators):
        self.c.execute(
            " UPDATE Constraints SET Operators = ? WHERE ConstraintID = ?", (Operators, ConstraintID))
        self.conn.commit()

    def AddReferenced(self, ObjectID):
        self.c.execute(
            " UPDATE Objects SET ReferencedInConstraint = 1 WHERE ObjectID = ?", (ObjectID,))
        self.conn.commit()

    def AddASTCol(self):
        self.c.execute("ALTER TABLE Constraints ADD AST text")

    def AddReferencesCol(self):
        self.c.execute("ALTER TABLE Constraints ADD ConstraintReferences text")

    def AddReferencedCol(self):
        self.c.execute("ALTER TABLE Objects ADD ReferencedInConstraint bit")

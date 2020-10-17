import sqlite3
import sqlalchemy


conn = sqlite3.connect('ThreeEyesDB.db')
c = conn.cursor()
c.execute("SELECT COUNT (DISTINCT ModelName) FROM Objects")
conn.commit()
result = c.fetchall()
print(result)
# IDs = []
# for row in result:
#     IDs.append(row[0])

# counter = 0
# for id in IDs:
#     c.execute("Select count(RelationID) FROM Relations WHERE ObjectID2=?",(id,))
#     result = c.fetchall()[0][0]
#     if result == 0:
#         print(str(id))
#         counter += 1
#         c.execute("DELETE FROM Objects WHERE ObjectID = ?", (id,))
# c.close()
# print(str(counter))


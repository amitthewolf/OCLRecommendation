from DAO import DAO
import re
from nltk.stem import WordNetLemmatizer


def parseObjectNames():
    objectNames_models = dao.getObjectNames()
    file = open("parsedObjectNames.txt", "a")

    for model,ObjectName in objectNames_models:
        if not ObjectName.isupper():
            ObjectName = re.sub(r"([A-Z])", r" \1", ObjectName).split()
            parsedObjectName = ' '.join(ObjectName)
            file.write(parsedObjectName + "\n")
        else:
            file.write(ObjectName + "\n")
    file.close()


dao = DAO()
parseObjectNames()
from DAO import DAO
import re
from nltk.stem import WordNetLemmatizer


def parseObjectNames():
    mdls_ctr = 1
    objectNames_models = dao.getObjectNames()
    file = open("parsedObjectNames.txt", "a")

    for model, ObjectName in objectNames_models:
        if mdls_ctr == model:
            mdls_ctr=mdls_ctr+1
        if not ObjectName.isupper():
            ObjectName = re.sub(r"([A-Z])", r" \1", ObjectName).split()
            parsedObjectName = ' '.join(ObjectName)
            file.write(str(model) + "#" + str(parsedObjectName) + "\n")
        else:
            file.write(str(model) + "#" + str(ObjectName) + "\n")

    print(mdls_ctr)
    file.close()


dao = DAO()
parseObjectNames()
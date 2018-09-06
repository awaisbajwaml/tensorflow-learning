import random
import pandas as pd

#mapping string to label class
TEXT_COLUMN_KEY = "text"
LABEL_COLUMN_KEY = "label"

randomHobbys = ["swimming", "knitting", "parachuting", "flying"]
professions = {
    "programmer" : {
        "hobbys": ["programming", "movies", "chess", "gaming", "love"],
        "label": 0
    },
    "craftsman" : {
        "hobbys": ["carving", "artwork", "bodybuilding", "gaming", "hiking", "love"],
        "label": 1
    },
    "reporter": {
        "hobbys": ["reading", "traveling", "love"],
        "label": 2
    }
}

def labelToString(labelValue):
    for key, value in professions.iteritems():
        tmpLabelValue = value["label"]
        if(tmpLabelValue == labelValue):
            return key
    return None

def createPandasData(amount, filterProfession=None):
    texts, labels = createRawData(amount, filterProfession)
    return pd.DataFrame({ TEXT_COLUMN_KEY: texts,
                          LABEL_COLUMN_KEY: labels})

def createRawData(amount, filterProfession=None):
    labels = []
    texts = []

    for i in range(amount):
        for profession in professions.keys():
            #if specified - filter out profession not macvhing the given profession
            if(filterProfession != None and profession != filterProfession):
                continue

            professionHobbys = []
            predefinedProfessionHobbys = professions[profession]["hobbys"]
            #for each profession - pick 3 hobbys which are typical and one random
            professionHobbys.extend(random.sample(predefinedProfessionHobbys, 3))
            professionHobbys.extend(random.sample(randomHobbys, 1))
            labels.append(professions[profession]["label"])
            texts.append(" ".join(professionHobbys))
    return texts, labels

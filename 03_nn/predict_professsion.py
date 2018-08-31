import csv
import random
import numpy as np;

def readData(csvPath):
    texts = []
    labels = []
    seed = 123

    with open(csvPath, 'rb') as csvfile:
        csvReader = csv.reader(csvfile, delimiter=';', quotechar='|')
        column = 0
        for row in csvReader:
            labelColumnVal = row[0]
            labelVal = 0
            # labels (classes) 0=programmer; 1=craftsman
            if(labelColumnVal == "programmer"):
                labels.append(0)
            else:
                labels.append(1)

            valueColumnVal = row[1]
            texts.append(valueColumnVal)

    #random.seed(seed)
    #random.shuffle(labels)

    #random.seed(seed)
    #random.shuffle(texts)

    return (np.array(labels), texts)


trainData = readData("professions_train.csv")
print(trainData)
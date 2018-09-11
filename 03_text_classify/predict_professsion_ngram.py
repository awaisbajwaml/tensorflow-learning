import random

import numpy as np
import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.python.keras import models
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import Dropout


#Simple demo for trying to predict a profession by the hobby of a person
#inspired by / taken from https://developers.google.com/machine-learning/guides/text-classification/step-1

# Vectorization parameters
# Range (inclusive) of n-gram sizes for tokenizing text.
NGRAM_RANGE = (1, 2)

# Limit on the number of features. We use the top 20K features.
TOP_K = 20000

# Whether text should be split into word or character n-grams.
# One of 'word', 'char'.
TOKEN_MODE = 'word'

# Minimum document/corpus frequency below which a token will be discarded.
MIN_DOCUMENT_FREQUENCY = 1

#mapping to label string to label class
LABEL_MAPPING = {"programmer": 0, "craftsman":1}

def createData(amount):
    seed = 123
    randomHobbys = ["swimming", "knitting", "parachuting", "flying"]
    professions = {
        "programmer" : {
            "hobbys": ["programming", "movies", "chess", "gaming", "love"]
        },
        "craftsman" : {
            "hobbys": ["carving", "artwork", "bodybuilding", "gaming", "hiking", "love"]
        }
    }

    labels = []
    texts = []

    for i in range(amount):
        for profession in professions.keys():
            professionHobbys = []
            predefinedProfessionHobbys = professions[profession]["hobbys"]
            #for each profession - pick 3 hobbys which are typical and one random
            professionHobbys.extend(random.sample(predefinedProfessionHobbys, 3))
            professionHobbys.extend(random.sample(randomHobbys, 1))

            labels.append(LABEL_MAPPING[profession])
            texts.append(" ".join(professionHobbys))


    #We do not want any information associated with the ordering of samples to influence the relationship between texts and labels.
    #pick the same seed for labels and data to keep their relationship
    random.seed(seed)
    random.shuffle(labels)

    random.seed(seed)
    random.shuffle(texts)
    return (texts, np.array(labels))

def vectorizeTextNgram(trainTexts, trainLabels, validationTexts):
    """Vectorizes texts as n-gram vectors. Implementation note: The training and validation text
    will use the same vocabulary so they will have the same shape. This will be important when you want to feed this data
    to the network

    1 text = 1 tf-idf vector the length of vocabulary of unigrams + bigrams.

    # Arguments
        trainTexts: list, training text strings.
        trainLabels: np.ndarray, training labels.
        validationTexts: list, validation text strings.

    # Returns
        vecTrainTexts, vecValidationTexts: vectorized training and validation texts
    """
    # Create keyword arguments to pass to the 'tf-idf' vectorizer.
    kwargs = {
        'ngram_range': NGRAM_RANGE,  # Use 1-grams + 2-grams.
        'dtype': 'int32',
        'strip_accents': 'unicode',
        'decode_error': 'replace',
        'analyzer': TOKEN_MODE,  # Split text into word tokens.
        'min_df': MIN_DOCUMENT_FREQUENCY,
        'max_features': TOP_K

    }
    vectorizer = TfidfVectorizer(**kwargs)

    # Learn vocabulary from training texts and vectorize training texts.
    vecTrainTexts = vectorizer.fit_transform(trainTexts)

    s = vecTrainTexts.shape[1:]
    print(type(s))
    print(vecTrainTexts.shape[1])

    # Vectorize validation texts.
    vecValidationTexts = vectorizer.transform(validationTexts)
    print(vecValidationTexts.shape[1])
    print(len(vectorizer.vocabulary_), vectorizer.vocabulary_)

    return vecTrainTexts, vecValidationTexts


def createMlpModel(numClasses, inputShape, dropoutRate=0.2, layers=2):
    #build our mlp model
    model = models.Sequential()
    #the input shape differs from the used vocabulary - so either supply it when predicting or normalize data
    model.add(Dropout(rate=dropoutRate, input_shape=inputShape))

    for _ in range(layers-1):
        model.add(Dense(units=64, activation='relu'))
        model.add(Dropout(rate=dropoutRate))

    model.add(Dense(units=numClasses, activation='softmax'))
    return model


#create input data 80% training and 20% validation
origTrainTexts, trainLabels = createData(100)
origValidationTexts, validationLabels = createData(20)

#vectorize the input data
vecTrainTexts, vecValidationTexts = vectorizeTextNgram(origTrainTexts, trainLabels, origValidationTexts)

# Create model and compile model with learning parameters.
model = createMlpModel(2, vecTrainTexts.shape[1:])
optimizer = tf.keras.optimizers.Adam(lr=1e-3)
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['acc'])

# Create callback for early stopping on validation loss. If the loss does
# not decrease in two consecutive tries, stop training.
callbacks = [tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=2)]

# Train and validate model.
history = model.fit(
    vecTrainTexts,
    trainLabels,
    epochs=1000,
    callbacks = callbacks,
    validation_data=(vecValidationTexts, validationLabels),
    verbose=2,  # Logs once per epoch.
    batch_size=128)

# Print results.
history = history.history
print('Validation accuracy: {acc}, loss: {loss}'.format(
    acc=history['val_acc'][-1], loss=history['val_loss'][-1]))

accuracyStats = model.evaluate(vecValidationTexts, validationLabels)
print(accuracyStats)


#pick some samples(hobbies) and try to predict the label (profession)
for i in range(5):
    record = vecValidationTexts[i]
    prediction = np.squeeze(model.predict(record))
    maxIndex = np.argmax(prediction);
    label = None
    for key,value in LABEL_MAPPING.iteritems():
        if(maxIndex == value):
            label = key

    print("hobbies", origValidationTexts[i])
    print("class", label)
    print(" ")

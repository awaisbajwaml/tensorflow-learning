import tensorflow as tf
from tensorflow.python.estimator import run_config
import tensorflow_hub as hub
import data_creator as dc

#text classification using the estimator module.
#This encapsulates most of the "low" level operation like preparing the data for the model,
#transforming the text into a dictionary and vetorizing the words into 128 floating point vector
#see module https://tfhub.dev/google/nnlm-en-dim128/1

#create input function for prediction data
def createPredictionInput(dataFrame):
    return tf.estimator.inputs.pandas_input_fn(dataFrame, shuffle=False)

#vectorize the pandas data frame
embedded_text_feature_column = hub.text_embedding_column(
    key=dc.TEXT_COLUMN_KEY,
    module_spec="https://tfhub.dev/google/nnlm-en-dim128/1")

config = run_config.RunConfig(model_dir="models")
estimator = tf.estimator.DNNClassifier(
    config=config,
    hidden_units=[500, 100],
    feature_columns=[embedded_text_feature_column],
    n_classes=len(dc.professions))

for profession in dc.professions:
    professionDataFrame = dc.createPandasData(1, profession)
    inputFunction = createPredictionInput(professionDataFrame)
    predictions = estimator.predict(inputFunction)

    maxProb = -1
    for prediction in predictions:
        tmpClassId = prediction["class_ids"][0]
        tmpProb = prediction["probabilities"][tmpClassId]

        if(tmpProb > maxProb):
            maxProb = tmpProb
            classLabel = dc.labelToString(tmpClassId);

    expectedLabel = dc.labelToString(professionDataFrame[dc.LABEL_COLUMN_KEY][0])
    predictionText = professionDataFrame[dc.TEXT_COLUMN_KEY][0]

    print("text to predict", predictionText)
    print("expected label", expectedLabel)
    print("actual label", classLabel)
    assert(expectedLabel == classLabel)







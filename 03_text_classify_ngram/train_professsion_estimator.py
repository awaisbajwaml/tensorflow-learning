import tensorflow as tf
from tensorflow.python.estimator import run_config
import tensorflow_hub as hub
import data_creator as dc

#text classification using the estimator module.
#This encapsulates most of the "low" level operation like preparing the data for the model,
#transforming the text into a dictionary and vetorizing the words into 128 floating point vector
#see module https://tfhub.dev/google/nnlm-en-dim128/1

#create train and test data
trainDataFrame = dc.createPandasData(1000)
testDataFrame = dc.createPandasData(200)

# Training input on the whole training set with no limit on training epochs.
train_input_fn = tf.estimator.inputs.pandas_input_fn(
    trainDataFrame, trainDataFrame[dc.LABEL_COLUMN_KEY], num_epochs=None, shuffle=True)
# Prediction on the whole training set.
predict_train_input_fn = tf.estimator.inputs.pandas_input_fn(
    trainDataFrame, trainDataFrame[dc.LABEL_COLUMN_KEY], shuffle=False)
# Prediction on the test set.
predict_test_input_fn = tf.estimator.inputs.pandas_input_fn(
    testDataFrame, testDataFrame[dc.LABEL_COLUMN_KEY], shuffle=False)

#the
embedded_text_feature_column = hub.text_embedding_column(
    key=dc.TEXT_COLUMN_KEY,
    module_spec="https://tfhub.dev/google/nnlm-en-dim128/1")

config = run_config.RunConfig(model_dir="models")
estimator = tf.estimator.DNNClassifier(
    config=config,
    hidden_units=[500, 100],
    feature_columns=[embedded_text_feature_column],
    n_classes=len(dc.professions),
    optimizer=tf.train.AdagradOptimizer(learning_rate=0.003))

# Training for 1,000 steps means 128,000 training examples with the default
# batch size. This is roughly equivalent to 5 epochs since the training dataset
# contains 25,000 examples.
estimator.train(input_fn=train_input_fn, steps=1000)

train_eval_result = estimator.evaluate(input_fn=predict_train_input_fn)
test_eval_result = estimator.evaluate(input_fn=predict_test_input_fn)

print("Training set accuracy: {accuracy}".format(**train_eval_result))
print("Test set accuracy: {accuracy}".format(**test_eval_result))

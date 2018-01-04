import pandas as pd
import numpy as np

import tensorflow as tf

FEATURES = ["PassengerId" ,"Pclass","Name","Sex","Age","SibSp","Parch","Ticket","Fare","Cabin","Embarked"]
LABEL = "Survived"

FILE_TRAIN = "titanic_train.csv"
FILE_TEST = "titanic_test.csv"
FILE_PREDICT = "titanic_predictions.csv"

def input_fn(file_path, repeat_count=1, shuffle=False):
  # define mapping function
  def decode_csv(line):
    parsed_line = tf.decode_csv(line, [[0], [0], [0], [""], ["unknown"], [0.], [0], [0], [""], [0.], [""], ["U"]])
    label = parsed_line[1]
    del parsed_line[1] # removes the label
    features = parsed_line
    d = dict(zip(FEATURES, features)), label
    return d
  dataset = (tf.data.TextLineDataset(file_path)
            .skip(1)
            .map(decode_csv))
  if shuffle:
    dataset = dataset.shuffle(buffer_size=256)
  dataset = dataset.repeat(repeat_count)
  dataset = dataset.batch(32)
  iterator = dataset.make_one_shot_iterator()
  batch_features, batch_labels = iterator.get_next()
  return batch_features, batch_labels

def predict_input_fn():
  # return to this in a moment
  prediction_set = pd.read_csv("titanic_predictions.csv")
  return tf.estimator.inputs.pandas_input_fn(
    x=prediction_set,
    num_epochs=1,
    shuffle=False)

  
def main():
  # continuous cols
  Pclass = tf.feature_column.numeric_column("Pclass")
  Age = tf.feature_column.numeric_column("Age")
  SibSp = tf.feature_column.numeric_column("SibSp")
  Parch = tf.feature_column.numeric_column("Parch")

  # categorial cols
  Sex = tf.feature_column.categorical_column_with_vocabulary_list(
      "Sex", ["female", "male", "unknown"])
  Embarked = tf.feature_column.categorical_column_with_vocabulary_list(
      "Embarked", ["C", "Q", "S", "U"])
  #buckets 
  age_buckets = tf.feature_column.bucketized_column(Age, 
         boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])

  feature_cols = [Pclass,
                  SibSp,
                  Parch,
                  tf.feature_column.indicator_column(Sex),
                  tf.feature_column.indicator_column(Embarked),
                  age_buckets]

  # set up classifier
  m = tf.estimator.DNNClassifier(
        model_dir="/tmp/titanicDNN",
        feature_columns=feature_cols,
        hidden_units=[10, 10],
        n_classes=2)
        
  # train model
  m.train(input_fn=lambda: input_fn(FILE_TRAIN, 8, True),
          steps=2000)
  # evaluate model
  evaluation_results = m.evaluate(input_fn=lambda: input_fn(FILE_TEST, 1, False))
  for key in  evaluation_results:
    print("    {} was:   {}".format(key, evaluation_results[key]))

  # try a prediction
  predictions = m.predict(input_fn=predict_input_fn())
  for idx, prediction in enumerate(predictions):
    survived = prediction["class_ids"][0]
    if survived:
      print("Survived")
    else:
      print("Died")
  
  
if __name__ == "__main__":
  main()

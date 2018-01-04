import pandas as pd
import numpy as np

import tensorflow as tf


FEATURES = ["Pclass", "Age", "SibSp", "Parch", "Sex", "Embarked"]
LABEL = "Survived"

# continuous cols
Pclass = tf.feature_column.numeric_column("Pclass")
Age = tf.feature_column.numeric_column("Age")
SibSp = tf.feature_column.numeric_column("SibSp")
Parch = tf.feature_column.numeric_column("Parch")

# categorial cols
Sex = tf.feature_column.categorical_column_with_vocabulary_list(
    "Sex", ["female", "male"])
Embarked = tf.feature_column.categorical_column_with_vocabulary_list(
    "Embarked", ["C", "Q", "S"])

feature_cols = [Pclass, Age, SibSp, Parch, Sex, Embarked]

def get_data():
  #loads the data into  dataframe then splits into training and test data
  df = pd.read_csv("titanic_data.csv")
  train=df.sample(frac=0.8,random_state=200)
  test=df.drop(train.index)
  return train, test
  


def input_fn(df, num_epochs, shuffle):
  # remove any NaNs
  df = df.dropna(how="any", axis=0)
  df_data = df[FEATURES]
  label = df[LABEL]
  return tf.estimator.inputs.pandas_input_fn(
    x=df_data,
    y=label,
    batch_size=100,
    num_epochs=num_epochs,
    shuffle=shuffle)

def predict_input_fn():
  prediction_set = pd.read_csv("titanic_predictions.csv")


  return tf.estimator.inputs.pandas_input_fn(
    x=prediction_set,
    num_epochs=1,
    shuffle=False)

  
def main():
  train, test = get_data()
  # set up classifier
  m = tf.estimator.LinearClassifier(
        model_dir="/tmp/titanic",
        feature_columns=feature_cols)
  # train model
  m.train(input_fn=input_fn(train, num_epochs=None, shuffle=True),
          steps=2000)
  # evaluate model
  accuracy_score = m.evaluate(input_fn=input_fn(test, num_epochs=1,
                                                shuffle=False))["accuracy"]
  print("\nAccuracy: {0:f}\n".format(accuracy_score))

  # try a prediction
  predictions = list(m.predict(input_fn=predict_input_fn()))
  
  predicted_classes = [p["classes"] for p in predictions]
  
  print(
      "Survival Predictions:    {}\n"
      .format(predicted_classes))


  

  
  
if __name__ == "__main__":
  main()

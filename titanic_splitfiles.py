import pandas as pd
import numpy as np

import tensorflow as tf

FEATURES = ["Pclass", "Age", "SibSp", "Parch", "Sex", "Embarked"]
LABEL = "Survived"


def get_data():
  #loads the data into  dataframe then splits into training and test data
  df = pd.read_csv("titanic_data.csv")
  train=df.sample(frac=0.8,random_state=200)
  test=df.drop(train.index)
  return train, test


def main():
  train, test = get_data()
  train.to_csv("titanic_train.csv", index=False)
  test.to_csv("titanic_test.csv", index=False)



  
if __name__ == "__main__":
  main()

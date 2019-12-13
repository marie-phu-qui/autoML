import pandas as pd

class AutoML :
    def __init__(self, dataset):
        self.dataset = dataset
    def preprocess(self):
        print(self.dataset.head())


data = pd.read_csv("../DATA/CRIME_NZ.csv")

automl = AutoML(data)
print(automl.preprocess())

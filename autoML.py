import pandas as pd
from sklearn.model_selection import train_test_split

class AutoML :
    def __init__(self, dataset, target):
        self.dataset = dataset
        self.target = target
    def reduce_data_size(self):
        data = self.dataset
    # Prendre toutes les variables de type int
        list_int64 = data.select_dtypes(include=['int64']).columns.tolist()
        for col in list_int64 :
            if data[col].max() <= 255 and data[col].min() > 0  :
                data[col] = data[col].astype('uint8')  # les uint8 rassemble les entiers positifs 0 à 255 
            elif data[col].max() <= 65535 and data[col].min() > 0  :
                data[col] = data[col].astype('uint16')
            elif data[col].max() <= 4294967295 and data[col].min() > 0  :
                data[col] = data[col].astype('uint32')
            elif data[col].max() <= 127 and data[col].min() >= -128  :
                data[col] = data[col].astype('int8')
            elif data[col].max() <= 32767 and data[col].min() >= -32768  :
                data[col] = data[col].astype('int16')
        # transfomer les colonnes str en variables catégorielles
        list_str = data.select_dtypes(include=['object']).columns.tolist()
        for col in list_str :
            data[col] = data[col].astype('category') 
        dict_dtypes = data.dtypes.to_dict()
        return dict_dtypes

    def preprocess(self):
        data = self.dataset
        target = self.target
        print(data[target][0])
        print(data.head())


data = pd.read_csv("CRIME_NZ.csv")
target = "Value"

automl = AutoML(data, target)
print(automl.reduce_data_size())

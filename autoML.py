import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class AutoML:

    def __init__(self, dataset, target):
        self.dataset = dataset
        self.target = target

    def reduce_data_size(self):
        ''' returns a dictionary of reduced dtypes to be applied on dataset '''

        data = self.dataset
    # Prendre toutes les variables de type int
        list_int64 = data.select_dtypes(include=['int64']).columns.tolist()
        for col in list_int64:
            if data[col].max() <= 255 and data[col].min() > 0:
                data[col] = data[col].astype('uint8')  # les uint8 rassemble les entiers positifs 0 à 255
            elif data[col].max() <= 65535 and data[col].min() > 0:
                data[col] = data[col].astype('uint16')
            elif data[col].max() <= 4294967295 and data[col].min() > 0:
                data[col] = data[col].astype('uint32')
            elif data[col].max() <= 127 and data[col].min() >= -128:
                data[col] = data[col].astype('int8')
            elif data[col].max() <= 32767 and data[col].min() >= -32768:
                data[col] = data[col].astype('int16')
        # transfomer les colonnes str en variables catégorielles
        list_str = data.select_dtypes(include=['object']).columns.tolist()
        for col in list_str:
            data[col] = data[col].astype('category')
        dict_dtypes = data.dtypes.to_dict()
        return dict_dtypes

    def preprocess(self):
        ''' returns a non categorical dataset '''

        data = self.dataset
        target = self.target

        y = data[target]
        X = data.loc[:, data.columns != target]

        # get_dummies sur variables catégorielles
        data_dummied = X.copy()
        data_dummied = pd.get_dummies(data_dummied, drop_first=True)

        # selectione les colonnes à scaler
        col_names = data_dummied.columns
        features = data_dummied[col_names]
        scaler = StandardScaler().fit_transform(features.values)
        # apply to data
        data_dummied[col_names] = scaler

        # utilise seulement les colonnes corrélées à plus de 80% à la target
        df_corr = pd.concat([data_dummied, y])
        correlation = df_corr.corr()
        print(data_dummied)
        print(correlation)
        return data_dummied

    def predict_reg(self):
        data = self.dataset
        target = self.target

    def predict_classifier(self):
        data = self.dataset
        target = self.target


data = pd.read_csv("CRIME_NZ.csv")
target = "Value"

automl = AutoML(data, target)
print(automl.preprocess())

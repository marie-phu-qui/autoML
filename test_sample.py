from autoML import AutoML
import pandas as pd

# content of test_sample.py


def inc(x):
    return x + 1


def test_answer():
    assert inc(3) == 4

# Create mock data set for regression case :


x_reg = [0, 1, 2, 3, 5]
y_reg = [0, 2, 4, 6, 10]
data = pd.DataFrame({'x': x_reg, 'y': y_reg})
automl = AutoML(data, 'y')


# importer AutoML reduce_data_size method : 


def test_reduce_exist():
    reduce_data_size = automl.reduce_data_size()
    assert len(reduce_data_size) != 0


def test_reduce_type():
    reduce_data_size = automl.reduce_data_size()
    assert type(reduce_data_size) == 'dict'

# importer AutoML preprocess method :


def test_preprocess():
    preprocess = automl.preprocess()
    object_type = preprocess.select_dtypes(include=['object']).columns.tolist()
    assert len(object_type) == 0

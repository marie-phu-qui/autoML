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
data = pd.DataFrame([x_reg, y_reg], columns=['x', 'y'])
automl = AutoML(data, 'y')


# importer AutoML reduce_data_size method :


reduce_data_size = AutoML.reduce_data_size()


def test_reduce():
    assert len(reduce_data_size) != 0

# importer AutoML preprocess method :


preprocess = AutoML.preprocess()


def preprocess():
    print(preprocess.dtypes())
    assert preprocess.dtypes()
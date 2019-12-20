from autoML import AutoML

# content of test_sample.py


def inc(x):
    return x + 1


def test_answer():
    assert inc(3) == 4

# importer AutoML reduce_data_size method :


data = [0, 1, 2, 3, 5]
reduce_data_size = AutoML.reduce_data_size(data)


def test_reduce():
    assert len(reduce_data_size) != 0

# importer AutoML preprocess method :


preprocess = AutoML.preprocess(data)


def preprocess():
    print(preprocess.dtypes())
    assert preprocess.dtypes()
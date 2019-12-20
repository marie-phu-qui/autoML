from autoML import AutoML

# content of test_sample.py


def inc(x):
    return x + 1


def test_answer():
    assert inc(3) == 4

# importer AutoML reduce_data_size method :


reduce_data_size = AutoML.reduce_data_size()


def test_reduce():
    assert len(reduce_data_size) != 0

# importer AutoML preprocess method :


preprocess = AutoML.preprocess()


def preprocess():
    assert len(preprocess) != 0

import numpy as np
import pandas as pd


class BatchGenerator(object):
    """generates batches"""

    def __init__(self, dataset, label, batch_size, input_size, is_shuffle=True):
        self._dataset = dataset
        self._label = label
        self._batch_size = batch_size
        self._cursor = 0
        self._input_size = input_size

        if is_shuffle:
            index = np.arange(len(self._dataset))
            np.random.shuffle(index)
            self._dataset = np.array(self._dataset)[index]
            self._label = np.array(self._label)[index]
        else:
            self._dataset = np.array(self._dataset)
            self._label = np.array(self._label)

    def next(self):
        if self._cursor + self._batch_size > len(self._dataset):
            self._cursor = 0
        """Generate a single batch from the current cursor position in the data."""
        batch_x = self._dataset[self._cursor: self._cursor + self._batch_size, :]
        batch_y = self._label[self._cursor: self._cursor + self._batch_size]
        self._cursor += self._batch_size
        return batch_x, batch_y


def get_data(essay_set: int):

    training_data = pd.read_excel("./data/training_set_rel3.xlsx")
    valid_data = pd.read_excel("./data/valid_set.xlsx")
    test_data = pd.read_csv("./data/test_set.tsv", sep="\t", encoding="Latin1")

    training_data = training_data[training_data["essay_set"] == essay_set]
    valid_data = valid_data[valid_data["essay_set"] == essay_set]
    test_data = test_data[test_data["essay_set"] == essay_set]

    train = training_data["essay"]
    train_label = training_data["domain1_score"]
    dev = valid_data["essay"]
    dev_label = valid_data["domain1_score"]
    test = test_data["essay"]
    test_label = test_data["domain1_score"]

    num_classes = len(training_data["domain1_score"].unique())

    return train, train_label, dev, dev_label, test, test_label, num_classes

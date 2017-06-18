# imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from os import path
import ID3

# constants
TRAIN_FILE = 'train.txt'
VALIDATION_FILE = 'validation.txt'
CSV_PARAMS = {'delim_whitespace': True, 'header': None}
FEATURE_VALUES = ['y', 'n', 'u']
TREES_DIR = 'trees'

def TREE_FILE(question, name):
    return path.join(TREES_DIR, 'q%d_%s.svg' % (question, name))

# read and tidy up the data
train_data = pd.read_csv(TRAIN_FILE, **CSV_PARAMS)
validation_data = pd.read_csv(VALIDATION_FILE, **CSV_PARAMS)

# change the last column's name to "label"
for data in [train_data, validation_data]:
    data.rename(columns={(data.shape[1]-1): 'label'}, inplace=True)

labels = pd.unique(train_data.label)
features = train_data.columns[:-1]

d = len(features)
d_values = np.array(range(d+1))

# initialize the ID3 class
T = ID3.ID3Classifier(labels, FEATURE_VALUES)

m_train = train_data.shape[0]
m_valid = validation_data.shape[0]
train_error = [None] * (d+1)
valid_error = train_error.copy()
for max_height in d_values:
    train_predictions = T.fit(train_data, max_height).predict(train_data)
    valid_predictions = T.predict(validation_data)
    train_error[max_height] = sum(train_predictions !=
                                  train_data.label) / m_train
    valid_error[max_height] = sum(valid_predictions !=
                                  validation_data.label) / m_valid
    T.save_to_file(TREE_FILE(question=2, name=str(max_height)))

plt.plot(d_values, train_error, label='training')
plt.plot(d_values, valid_error, label='validation')
plt.xlabel('maximal tree height (d)')
plt.ylabel('error')
plt.legend()
plt.show()

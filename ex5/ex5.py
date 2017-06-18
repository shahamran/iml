# imports
import pandas as pd
import numpy as np
from ID3 import ID3
import matplotlib.pyplot as plt
from anytree.iterators import PreOrderIter
from anytree.dotexport import RenderTreeGraph
import pptree

# constants
TRAIN_FILE = 'train.txt'
VALIDATION_FILE = 'validation.txt'
CSV_PARAMS = {'delim_whitespace': True, 'header': None}
FEATURE_VALUES = ['y', 'n', 'u']

# read and tidy up the data
train_data = pd.read_csv(TRAIN_FILE, **CSV_PARAMS)
validation_data = pd.read_csv(VALIDATION_FILE, **CSV_PARAMS)
# change the last column's name to "label"
for data in [train_data, validation_data]:
    data.rename(columns={(data.shape[1]-1): 'label'}, inplace=True)
# map possible label values to the set {0, ... , d-1}
labels = pd.unique(train_data.label)
num_to_label = {i: label for i, label in enumerate(labels)}
label_to_num = {label: i for i, label in enumerate(labels)}
train_data.label = train_data.label.map(label_to_num)
validation_data.label = validation_data.label.map(label_to_num)
# num_questions == number of "features"
features = train_data.columns[:-1]
d = len(features)

T = ID3(feature_values=FEATURE_VALUES, label_values=np.arange(len(labels)))

m_train = train_data.shape[0]
m_valid = validation_data.shape[0]
train_error = [None] * (d+1)
valid_error = train_error.copy()
for depth in range(d+1):
    T.train(train_data, depth)
    train_error[depth] = sum(T.predict(train_data) !=
                             train_data.label) / m_train
    valid_error[depth] = sum(T.predict(validation_data) !=
                             validation_data.label) / m_valid
plt.plot(range(d+1), train_error, label='training')
plt.plot(range(d+1), valid_error, label='validation')
plt.xlabel('maximal height of the tree (d)')
plt.ylabel('error')
plt.legend()
plt.show()

# imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ID3
from anytree.dotexport import RenderTreeGraph
from os import path

# constants
TRAIN_FILE = 'train.txt'
VALIDATION_FILE = 'validation.txt'
CSV_PARAMS = {'delim_whitespace': True, 'header': None}
FEATURE_VALUES = ['y', 'n', 'u']
TREES_DIR = 'trees'

def TREE_FILE(name):
    return path.join(TREES_DIR, name)

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
d_values = np.array(range(d+1))

m_train = train_data.shape[0]
m_valid = validation_data.shape[0]
train_error = [None] * (d+1)
valid_error = train_error.copy()
for max_height in d_values:
    T = ID3.train(train_data, max_height)
    train_error[max_height] = sum(ID3.predict(T, train_data) !=
                                  train_data.label) / m_train
    valid_error[max_height] = sum(ID3.predict(T, validation_data) !=
                                  validation_data.label) / m_valid
    # print('Tree of height <= %d' % depth)
    # ID3.show(T)


    # print('\n' + 'max height = %d' % depth)
    # print('nodes = %d' % T.root._nodes)
    # print('height = %d' % T.root._height)
    # print('root = %s' % T.root.name)
# pptree.print_tree(T.root, '_children')
def get_answer(node, child):
    return 'label=' + str(child.value)
def node_name(node):
    if node.name in num_to_label:
        name = num_to_label[node.name]
    else:
        name = node.name
    return '%s (%d)' % (name, node.depth)
# RenderTreeGraph(T, nodenamefunc=node_name, edgeattrfunc=get_answer).to_picture(
#     'test.png')
plt.plot(d_values, train_error, label='training')
plt.plot(d_values, valid_error, label='validation')
plt.xlabel('maximal height of the tree (d)')
plt.ylabel('error')
plt.legend()
plt.show()

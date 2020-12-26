
import numpy as np
import csv
from tqdm import trange, tqdm
import math
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import sklearn
import sklearn.metrics as metrics
from tqdm import tqdm
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
from sklearn.metrics import roc_auc_score
import io
import pydot

acs = []
data = np.load('../neural-network/sampled_kura15new.npy', allow_pickle=True)
df1 = pd.read_csv('../neural-network/sync (52).csv', header=None)
y = df1.to_numpy()
its = [390 - (15*i) for i in range(25)]
for it in tqdm(its):
	input_size = 390 - it + 10
	print('Using first ' + str((390 - it) / 15) + ' iterations')
	new_data = []
	print('reformatting data')
	for i in trange(data.shape[0]):
		a = np.delete(data[i], list(range(390 - it, 390)))
		new_data.append(a)
	new_data = np.array(new_data)

	np.reshape(new_data, newshape=(50000, input_size))
	print('assigning data')
	datat = new_data

	train_data, test_data, train_labels, test_labels = train_test_split(datat, y, test_size=0.10, random_state=57109)
	train_labels = np.ravel(train_labels)
	test_labels = np.ravel(test_labels)

	rf = RandomForestClassifier(random_state=0, n_estimators=100,  max_features='sqrt')
	rf.fit(train_data, (train_labels))
	print('Predicting on test set')
	prediction = rf.predict(test_data)
	accuracy = accuracy_score(test_labels, prediction)
	print('Accuracy of model with iteration(s) ' + str(it) + ' is: ' + str(accuracy))
	acs.append(accuracy)
print(acs)
with open('kura15rerunRF.npy', 'wb') as npyf:
    #save numpy binary file 
    np.save(npyf, np.array(acs))


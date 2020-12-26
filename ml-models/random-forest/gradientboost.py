from scipy import stats, io
import numpy as np
import csv
from tqdm import trange, tqdm
import math
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import GradientBoostingClassifier
import matplotlib.pyplot as plt
import sklearn.metrics as metrics


data = np.load('../neural-network/sampled_kura15new.npy', allow_pickle=True)

df1 = pd.read_csv('../neural-network/sync (52).csv', header=None)
y = df1.to_numpy()
acs = []
fts = 390
nodes = 15
its = [fts - nodes*i for i in range(25)]
for it in tqdm(its):
	input_size = fts + 10 - it
	print('Using first ' + str((fts - it) / nodes) + ' iterations')
	
	new_data = []
	print('reformatting data')
	for i in trange(data.shape[0]):
		a = np.delete(data[i], list(range(fts - it, fts)))
		new_data.append(a)
	new_data = np.array(new_data)

	np.reshape(new_data, newshape=(50000, input_size))
	print('assigning data')
	datat = new_data
	
	train_data, test_data, train_labels, test_labels = train_test_split(datat, y, test_size=0.10, random_state=57109)
	
	train_labels = np.ravel(train_labels)
	test_labels = np.ravel(test_labels)

	gb = GradientBoostingClassifier(learning_rate=0.4, random_state=57109)
	print('Fitting')
	tqdm(gb.fit(train_data, train_labels))
	print('Predicting')
	prediction = gb.predict(test_data)
	accuracy = accuracy_score(test_labels, prediction)
	print('Accuracy of model is:', str(accuracy))
	acs.append(accuracy)
with open(str(nodes) +'kura15rerunGB.npy', 'wb') as npyf:
	np.save(npyf, np.array(acs))

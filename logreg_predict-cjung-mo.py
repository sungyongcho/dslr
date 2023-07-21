import numpy as np
import pandas as pd
import sys, os, pickle
from sklearn.metrics import accuracy_score
from logreg_train import normalization, predict_

def load_data(path):
	#print(f"path: {path}")
	try:
		df = pd.read_csv(path, index_col=0)
	except:
		print("Invalid file error.")
		sys.exit()
	#print("df shape:", df.shape)
	features = df.columns.tolist()

	return (df, features)

def modify_mapping_dict(mapping, house):
	if 0 <= house <= 4:
		keys_list = list(mapping.keys())
		mapping[keys_list[house]] = 1
	else:
		print("Invalid input: house must be in the range [0, 4] inclusive")

def map_houses(data):
	mapping = {'Ravenclaw': 0, 'Slytherin': 1, 'Gryffindor': 2, 'Hufflepuff': 3}
	mapped_data = data['Hogwarts House'].replace(mapping).values
	return mapped_data

def map_one_house(data, house):
	mapping = {'Ravenclaw': 0, 'Slytherin': 0, 'Gryffindor': 0, 'Hufflepuff': 0}
	modify_mapping_dict(mapping, house)
	mapped_data = data['Hogwarts House'].replace(mapping).values
	return mapped_data

def map_one_house(df, house):
	mapping = {0: 0, 1: 0, 2: 0, 3: 0}
	#print(df)
	modify_mapping_dict(mapping, house)
	mapped_data = df['Hogwarts House'].replace(mapping).values
	#print(mapped_data)
	return mapped_data

def display_accuracy(df, y_pred, house):
	#data, features = load_data('dataset_truth.csv')
	y_test = map_one_house(df, house)
	#print('display:', y_test.shape, y_pred.shape)
	accuracy = accuracy_score(y_test, y_pred)
	print('Accuracy:', accuracy)

def load_models():
	# Load the models from the pickle file
	filename = "models.pickle"
	with open(filename, 'rb') as file:
		models = pickle.load(file)

	classifiers = []
	for house, classifier in models.items():
		print(f"\nhouse:{house}")
		for classifier_data in classifier:
			print(classifier_data)

	return models

def best_hypothesis(test_df, models):
	for house, classifier in models.items():
		print(f"\nhouse:{house}")
		best_alpha = 0.0
		best_max_iter = float('inf')
		for classifier_data in classifier:
			if (classifier_data['alpha'] > best_alpha):
				best_alpha = classifier_data['alpha']
			if (classifier_data['max_iter'] < best_max_iter):
				best_max_iter = classifier_data['max_iter']
		print(best_alpha, best_max_iter)

def evaluate_models(test_df, models):
	for house, classifier in models.items():
		print(f"\nhouse:{house}")
		for classifier_data in classifier:
			probability = predict_(test_df.values[:,:-1], classifier_data['thetas'])
			#print('prob:', probability)
			binary_predictions = (probability >= 0.5).astype(int)
			#print('bi_prob:', binary_predictions)
			display_accuracy(test_df, binary_predictions, house)

if __name__ == "__main__":
	df, features = load_data('./datasets/dataset_test.csv')
	truth_df, _ = load_data('./datasets/dataset_truth.csv')

	df= df.drop(columns=['Arithmancy', 'Potions',
 							'Care of Magical Creatures', 'Hogwarts House'])
	mapping = {'Ravenclaw': 0, 'Slytherin': 1, 'Gryffindor': 2, 'Hufflepuff': 3}

	x = df.select_dtypes(include='number')
	normalized_x, data_min, data_max = normalization(x)
	y = truth_df.replace(mapping)
	#new_data = np.column_stack((normalized_x, y))
	combined_df = pd.concat([normalized_x, y], axis=1)
	combined_df.dropna(inplace=True)
	print('combiend df shape:', combined_df.values.shape)

	models = load_models()
	evaluate_models(combined_df, models)
	#best_hypothesis(combined_df, models)


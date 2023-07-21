import pandas as pd
import numpy as np
import sys, os, itertools, pickle
from sklearn.metrics import accuracy_score
from TinyStatistician import TinyStatistician as Tstat
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from evaluate import display_accuracy
from logreg_train import predict_, extract_train_data

def load_data(path):
	print(f"path: {path}")
	try:
		df = pd.read_csv(path, index_col=0)
	except:
		print("Invalid file error.")
		sys.exit()
	print("df shape:", df.shape)
	features = df.columns.tolist()

	return (df, features)

def normalization(data):
	data_min = np.min(data, axis=0)
	data_max = np.max(data, axis=0)
	normalized_data = (data - data_min) / (data_max - data_min)
	return normalized_data, data_min, data_max

def denormalization(normalized_data, data_min, data_max):
	x = normalized_data * (data_max - data_min)
	denormalized_data = normalized_data * (data_max - data_min) + data_min
	return denormalized_data

def denormalize_thetas(thetas, data_max, data_min):
	# Recover the slope of the line
	slope = thetas[1] * (data_max[1] - data_min[1]) / (data_max[0] - data_min[0])
	# Recover the intercept of the line
	intercept = thetas[0] * (data_max[1] - data_min[1]) + data_min[1] - slope * data_min[0]
	denormalized_thetas = np.array([intercept, slope]).reshape(-1, 1)
	return denormalized_thetas

def label_data(y, house):
	y_ = np.zeros(y.shape)
	y_[np.where(y == int(house))] = 1
	y_labelled = y_.reshape(-1, 1)
	#print("y_labelled shape:", y_labelled.shape)
	#print("y_labelled[:5]:", y_labelled[:5])
	return y_labelled

def data_spliter_by(x, y, house):
	#print("y:", y, "house:", house)
	y_ = np.zeros(y.shape)
	y_[np.where(y == (house))] = 1
	y_labelled = y_.reshape(-1, 1)
	#print("y_labelled shape:", y_labelled.shape)
	#print("y_labelled[:5]:", y_labelled[:5])
	return train_test_split(x, y_labelled, test_size=0.2, random_state=42)

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

def evaluate(thetas, alpha):
    y_pred = pred['Hogwarts House']
    y_true = true['Hogwarts House']
    print(accuracy_score(y_true, y_pred))
    print(accuracy_score(y_true, y_pred, normalize=False))

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
	predictions = np.zeros(test_df.values.shape[0])

	for house, classifier in models.items():
		print(f"\nhouse:{house}")
		for classifier_data in classifier:
			probability = predict_(test_df.values[:,:-1], classifier_data['thetas'])
			#print('prob:', probability)
			binary_predictions = (probability >= 0.5).astype(int)
			#print('bi_prob:', binary_predictions)
			#predictions[np.where(binary_predictions == 1)] = 1 #house
			#print('pred:', predictions)
			display_accuracy(test_df, binary_predictions, house)

if __name__ == "__main__":
	df, features = load_data('./dataset_test.csv')
	truth_df, _ = load_data('./dataset_truth.csv')
	#data = extract_train_data(df)
	numerical_df = df.select_dtypes(include='number')
	print("numerical_df shape:", numerical_df.shape)

	numerical_features = numerical_df.columns.tolist()
	#print(f"numerical_features: {numerical_features}")

	numerical_data = numerical_df.values
	#print("numerical_data:", numerical_data)
	print("numerical_data shape:", numerical_data.shape)

	data = df.copy()
	data = data.drop(columns=['Arithmancy', 'Potions',
 							'Care of Magical Creatures', 'Hogwarts House'])
	#data.dropna(inplace=True)
	mapping = {'Ravenclaw': 0, 'Slytherin': 1, 'Gryffindor': 2, 'Hufflepuff': 3}

	x = data.select_dtypes(include='number')
	normalized_x, data_min, data_max = normalization(x)
	y = truth_df.replace(mapping)
	#new_data = np.column_stack((normalized_x, y))
	combined_df = pd.concat([normalized_x, y], axis=1)
	combined_df.dropna(inplace=True)


	models = load_models()
	print('combiend df shape:', combined_df.values.shape)
	#print('norm_x:', normalized_x[:4])
	evaluate_models(combined_df, models)
	best_hypothesis(combined_df, models)


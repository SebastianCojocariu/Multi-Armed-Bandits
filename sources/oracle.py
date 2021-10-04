from sklearn import *
from sklearn.ensemble import *
from sklearn.tree import *
from sklearn.neural_network import *
from sklearn.svm import *

from xgboost import *
from lightgbm import *

from collections import Counter
import xgboost as xgb
import numpy as np

# Base class to store an oracle
class Oracle():
	def __init__(self, args):
		self.args = args
		self.model = None


	def fit(self, X, y):
		# if there is only one label available, return False (as we cannot fit the data)
		if len(Counter(y)) < 2:
			return False

		self.model = self.initialize_oracle()
		self.model.fit(X=X, y=y)
		return True


	def predict(self, X):
		X = np.asarray(X)
		if len(X.shape) == 1:
			X = np.reshape(X, (1, -1))

		# if the model is not fitted, return random predictions
		if self.model is None:
			predictions = np.random.rand(X.shape[0])
		# else, use the model to predict.
		else:		
			predictions = self.model.predict_proba(X)[:, 1]
		
		predictions = np.squeeze(predictions)
		
		return predictions


	def initialize_oracle(self):
		assert "class_model_name" in self.args and "model_args" in self.args, "To use an oracle, pass class_model_name and model_args!"
		oracle = globals()[self.args["class_model_name"]](**self.args["model_args"])
		assert oracle is not None, "Oracle could not be instantiated! Check the model_args again!"
		return oracle

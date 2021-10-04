import numpy as np
import matplotlib.pyplot as plt
import ast
import shap
import xgboost as xgb

from sklearn.multioutput import MultiOutputClassifier
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance


class YahooReader():
	def __init__(self, file_path, eventssize_limit=float("Infinity")):
		self.file_path = file_path
		self.eventssize_limit = eventssize_limit
		self.articles_dict = {} # <--- dictionary that stores as keys the available articles
		self.initial_features_dict = {i + 2: i for i in range(135)} # <--- initial features mappings. (+2 is added because the important features start from 2)
		self.reduced_features_dict = {} # <--- dictionary to store the reduced features mappings.
		self.contexts_dict = {} # <--- dictionary that stores the contexts that are used (for Exploratory Data Analysis)
		
		self.__explore()


	def create_event_generator(self, sparse_context, reduce_context_size):		
		with open(self.file_path, "r") as f:
			lines = f.read().splitlines() # for better RAM usage we should use readline() in a while loop
			lines = lines[:min(self.eventssize_limit, len(lines))]
			
			for i, line in enumerate(lines):
				partitions = [partition.split() for partition in line.split("|")]
				
				timestamp = partitions[0][0]
				target_article = partitions[0][1]
				click_flag = int(partitions[0][2]) 
				
				context = np.asarray([int(feature) for feature in partitions[1][2:]])
				context.sort()
				
				if sparse_context:
					pass
				else: 
					context = self.prepare_context(context=context, reduce_context_size=reduce_context_size)

				articles_pool = [article for [article] in partitions[2: ]]
				
				yield {"context": context, "target_arm": target_article, "reward": click_flag, "available_arms": articles_pool}


	def prepare_context(self, context, reduce_context_size):
		if reduce_context_size and len(self.reduced_features_dict) > 0:
			mapping_dict = self.reduced_features_dict
		else:
			mapping_dict = self.initial_features_dict

		output = np.zeros(len(mapping_dict))
		for feature in context:
			output[mapping_dict[feature]] = 1
		
		return output


	def __explore(self):
		event_generator = self.create_event_generator(sparse_context=True, reduce_context_size=False)
		no_events = 0
		no_recomended_articles_per_event = []

		while(True):
			try:
				event = next(event_generator)
			except StopIteration:
				break

			no_events += 1
			context, target_article, reward, articles_pool = event["context"], event["target_arm"], event["reward"], event["available_arms"]
			no_recomended_articles_per_event.append(len(articles_pool))

			for feature in context:
				if feature not in self.reduced_features_dict:
					self.reduced_features_dict[feature] = len(self.reduced_features_dict)

			if target_article not in self.articles_dict:
				self.articles_dict[target_article] = len(self.articles_dict)
			for article in articles_pool:
				if article not in self.articles_dict:
					self.articles_dict[article] = len(self.articles_dict)

			######### Statistics #########
			context_str = np.array2string(context, precision=int, separator=',', suppress_small=True)
			
			if context_str not in self.contexts_dict:
				self.contexts_dict[context_str] = {}

			if target_article not in self.contexts_dict[context_str]:
		 		self.contexts_dict[context_str][target_article] = {0: 0, 1: 0} # <--- rewards histogram

			self.contexts_dict[context_str][target_article][reward] += 1
							
			if no_events % 10000 == 0:
				print("[Exploring] Finished preprocessing {}-th event".format(no_events), end="\r")
		
		print()
		print("###### Total Dataset Size: {} #######".format(no_events))
		print("###### Average number of articles per event: {} #######".format(np.mean(no_recomended_articles_per_event)))
		print("###### Number of used features: {} #######".format(len(self.reduced_features_dict)))
		print("###### Number of different articles: {} #######".format(len(self.articles_dict)))
		print()


	def exploratory_data_analysis(self):
		no_confusing, no_positive_reward, no_negative_reward, no_multilabel, no_nonmultilabel = 0, 0, 0, 0, 0
		
		for context_str in self.contexts_dict:
			number_of_arms_with_positive_vote = 0
			
			for arm in self.contexts_dict[context_str]:
				if self.contexts_dict[context_str][arm][0] >= 1 and self.contexts_dict[context_str][arm][1] >= 1:
					no_confusing += 1
				elif self.contexts_dict[context_str][arm][0] >= 1:
					no_negative_reward += 1
				else:
					no_positive_reward += 1

				if self.contexts_dict[context_str][arm][1] >= 1:
					number_of_arms_with_positive_vote += 1
			
			if number_of_arms_with_positive_vote >= 2:
				no_multilabel += 1
			else:
				no_nonmultilabel += 1
		
		print("#### No contexts available: {} ####".format(len(self.contexts_dict)))
		print("#### Confusing samples: {} ####".format(no_confusing))
		print("#### No positive samples: {} ####".format(no_positive_reward))
		print("#### No negative samples: {} ####".format(no_negative_reward))
		print("#### MultiLabel pairs: {} ####".format(no_multilabel))
		print("#### SingleLabel pairs: {} ####".format(no_nonmultilabel))
		print()


	def collect_events_for_article(self, article_id):
		X, y = [], []
		for context_str in self.contexts_dict:
			negative_label, no_positive, positive_label = 0, 0, 0
			for current_article in self.contexts_dict[context_str]:
				histogram = self.contexts_dict[context_str][current_article]
				
				# Inconsistent case (the features are not descriptive enough and we have different rewards for the same context)
				if (histogram[0] >= 1 and histogram[1] >= 1):
					continue

				if histogram[1] >= 1:
					if current_article == article_id:
						positive_label = 1
						no_positive = histogram[1] # <-- give a higher weight for the article_id
					else:
						negative_label = 1

			# If there is at least one label set to 1
			if not (positive_label == 0 and negative_label == 0):
				context = self.prepare_context(context=ast.literal_eval(context_str), reduce_context_size=True)
				label = [negative_label, positive_label]
				# If the context provides a reward of 1 only for the article_id, add it multiple times
				# (because of the imbalance)
				if no_positive > 0:
					for _ in range(no_positive):
						X.append(context)
						y.append(label)
				# Otherwise, add it only one time
				else:
					X.append(context)
					y.append(label)

		return (np.asarray(X), np.asarray(y))
		

	def plot_most_important_features_per_article(self, article_id, shap_flag=False, top_features=25):
		if article_id not in self.articles_dict:
			raise Exception("Unknown article id: {}".format(article_id))

		X, y = self.collect_events_for_article(article_id=article_id)

		print("Number of samples: {}".format(len(X)))
		freq = Counter([tuple(a) for a in y])
		print("Frequency: {}".format(freq))
		
		positive_class, negative_class = 0, 0
	
		for key in freq:
			if key[0] == 1:
				negative_class += freq[key]
			if key[1] == 1:
				positive_class += freq[key]
		
		imbalance = negative_class // positive_class
		print("Imbalance: {}".format(imbalance))

		idx2feature_dict = {self.reduced_features_dict[key]: key for key in self.reduced_features_dict}

		#clf = RandomForestClassifier(random_state=0, class_weight="balanced")
		clf = xgb.XGBClassifier(objective='binary:logistic', scale_pos_weight=imbalance)
		clf = MultiOutputClassifier(clf)
		clf.fit(X, y)

		'''
		feat_impts = [] 
		for estimator in clf.estimators_:
			feat_impts.append(estimator.feature_importances_)

		importances = np.mean(feat_impts, axis=0)
		importances /= importances.sum()
		print(importances)
		print(len(importances))

		############ SHAP ############
		if shap_flag:
			# Doesn't currently work this way because of the multilabel approach
			explainer = shap.Explainer(clf)
			shap_values = explainer(X)
			shap.waterfall_plot(explainer.base_values[0], shap_values[0][0], X[0])
			#shap.waterfall_plot(shap_values[0])

		#importances = clf.feature_importances_

		# Choose the top_features based on importance, in descending order
		col_sorted_by_importance = importances.argsort()[-top_features:][::-1]
		importances = importances[col_sorted_by_importance]
		feature_names = ["Feature {}".format(str(idx2feature_dict[idx])) for idx in col_sorted_by_importance]

		# Plot
		clf_importances = pd.Series(importances, index=feature_names)
		fig, ax = plt.subplots()
		clf_importances.plot.bar(ax=ax)
		ax.set_title("MDI Technique")
		ax.set_ylabel("Feature Importance")
		fig.tight_layout()
		plt.show()
		'''
	
		############ Feature Permutation ############
		result = permutation_importance(clf, X, y, n_repeats=10, random_state=42, n_jobs=2)
		importances = result.importances_mean
		importances /= importances.sum()
		
		# Choose the top_features based on importance, in descending order
		col_sorted_by_importance = importances.argsort()[-top_features:][::-1]
		importances = importances[col_sorted_by_importance]
		feature_names = ["Feature {}".format(str(idx2feature_dict[idx])) for idx in col_sorted_by_importance]

		# Plot
		clf_importances = pd.Series(importances, index=feature_names)
		fig, ax = plt.subplots()
		clf_importances.plot.bar(ax=ax)
		ax.set_title("Permutation Technique")
		ax.set_ylabel("Feature Importance")
		fig.tight_layout()
		plt.show()

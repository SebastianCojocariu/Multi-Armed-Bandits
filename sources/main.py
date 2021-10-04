import sys
import numpy as np
import matplotlib.pyplot as plt
import json
import os
sys.path.append(".")

from itertools import cycle
from scipy.signal import savgol_filter

# Context-based policies
from contextual_epsilon_greedy import ContextualEpsilonGreedy
from contextual_explore_exploit import ContextualExploreExploit
from contextual_adaptive_greedy import ContextualAdaptiveGreedy
from lin_ucb import LinUCB
from bootstrapped_ucb import BootstrappedUCB
from bootstrapped_thompson import BootstrappedThompson

# Context-free policies
from random_policy import RandomPolicy
from epsilon_greedy import EpsilonGreedy
from explore_exploit import ExploreExploit
from thompson import Thompson
from ucb import UCB 

from yahoo_reader import YahooReader


def smooth(y):
    y_smooth = savgol_filter(y, 51, 3)
    return y_smooth


def custom_plot(lines_args, save_path="", title=""):
	COLORS = cycle(["r", "g", "b", "c", "m", "y", "k", "tab:blue", "tab:orange", "tab:green", "tab:purple", "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan"])
	fig, ax = plt.subplots(1)
	
	min_x, max_x = float("Infinity"), 0
	max_y = 0
	for line_args in lines_args:
		Y = smooth(line_args["total_rewards"])
		X = np.arange(Y.shape[0])
		ax.plot(X, Y, label=line_args["policy_name"], color=next(COLORS))
		max_y = max(max_y, Y.max())
		min_x = min(min_x, Y.shape[0])
		max_x = max(max_x, Y.shape[0])

	ax.set_title(title)
	ax.set_xlabel("Number of processed events (offline policy)")
	ax.set_ylabel("Click Through Rate")
	
	ax.set_xlim(left=0, right=max_x)
	ax.set_ylim(bottom=0, top=max_y)

	ax.legend()
	if save_path not in (None, ""):
		plt.savefig(fname=save_path, dpi=500)
	plt.show()


def test_contextual_free(save_path="", limit=10000):
	reader = YahooReader(file_path="../dataset/ydata-fp-td-clicks-v2_0.20111011", eventssize_limit=limit)
	trainer1 = RandomPolicy(policy_args={})
	trainer2 = EpsilonGreedy(policy_args={"p": 0.2, "decay": 1.0})
	trainer3 = ExploreExploit(policy_args={"breakpoint": 5000, "p": 0.0})
	trainer4 = Thompson(policy_args={"p": 0.0})
	trainer5 = UCB(policy_args={"confidence_level": 2.0, "p": 0.0})

	if save_path not in (None, ""):
		os.makedirs(save_path, exist_ok=True)
	
	results = []
	for trainer in [trainer1, trainer2, trainer3, trainer4, trainer5]:
		intermediate_result = trainer.fit(event_generator=reader.create_event_generator(sparse_context=False, reduce_context_size=True), debug=True)
		results.append(intermediate_result)
		if save_path not in (None, ""):
			with open(os.path.join(save_path, "{}.npy".format(trainer.policy_name)), 'wb') as f:
				np.save(f, intermediate_result["total_rewards"])
		print("Finished training: {}".format(trainer.policy_name))
	
	if save_path not in (None, ""):
		custom_plot(results, save_path=os.path.join(save_path, "comparison_contextual_free"), title="Context-Free Policies")
	else:
		custom_plot(results, save_path=None, title="Context-Free Policies")


def test_contextual(model_args, save_path="", limit=10000):
	reader = YahooReader(file_path="../dataset/ydata-fp-td-clicks-v2_0.20111011", eventssize_limit=limit)

	trainer1 = BootstrappedThompson(policy_args={"m": 3, "threshold_retrain_percent": 0.2, "p": 0.05, "states_heuristics": True},
						 	  model_args=model_args)
	trainer2 = ContextualExploreExploit(policy_args={"breakpoint": 5000, "threshold_retrain_percent": 0.2, "p": 0.05, "states_heuristics": True},
						 		  model_args=model_args)
	trainer3 = ContextualAdaptiveGreedy(policy_args={"threshold": 0.8, "decay": 0.9997, "threshold_retrain_percent": 0.2, "p": 0.05, "states_heuristics": True},
								  model_args=model_args)
	trainer4 = ContextualEpsilonGreedy(policy_args={"p": 0.2, "decay": 0.9997, "threshold_retrain_percent": 0.2, "states_heuristics": True},
								 model_args=model_args)
	trainer5 = BootstrappedUCB(policy_args={"percentile": 50, "m": 3, "threshold_retrain_percent": 0.2, "p": 0.05, "states_heuristics": True},
						 model_args=model_args)
	trainer6 = LinUCB(policy_args={"alpha": 1.0, "p": 0.05})

	if save_path not in (None, ""):
		os.makedirs(save_path, exist_ok=True)
	
	results = []
	for trainer in [trainer1, trainer2, trainer3, trainer4, trainer5, trainer6]:
		intermediate_result = trainer.fit(event_generator=reader.create_event_generator(sparse_context=False, reduce_context_size=True), debug=True)
		results.append(intermediate_result)
		if save_path not in (None, ""):
			with open(os.path.join(save_path, "{}.npy".format(trainer.policy_name)), 'wb') as f:
				np.save(f, intermediate_result["total_rewards"])
		print("Finished training: {}".format(trainer.policy_name))
	
	if save_path not in (None, ""):
		custom_plot(results, save_path=os.path.join(save_path, "comparison_{}".format(model_args)), title="Context Based Policies")
	else:
		custom_plot(results, save_path=None, title="Context Based Policies")


if __name__ == "__main__":
	'''
	test_contextual_free(save_path="./results") #<--- for testing purposes
	test_contextual(model_args={"class_model_name": "SVC", "model_args": {"probability": True}}, save_path="./results") #<--- for testing purposes
	'''

	config_file = json.load(open("config_file.json", "r"))
	trainer = globals()[config_file["policy_name"]](**config_file["args"])
	reader = YahooReader(file_path=config_file["file_path"], eventssize_limit=config_file["limit_size"])
	intermediate_result = trainer.fit(event_generator=reader.create_event_generator(sparse_context=False, reduce_context_size=True), debug=True)

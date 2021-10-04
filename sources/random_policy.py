from contextual_free_bandit import *

class RandomPolicy(ContextualFreeBandit):
	def __init__(self, policy_args={}):
		super().__init__(policy_args=policy_args)
		self.policy_name = "RandomPolicy"

	
	def compute_score(self, arm):
		return 1.0 # <--- return the same score, irrespective of the arm


	def choose_arm(self, event):
		return random.choice(event["available_arms"]) 

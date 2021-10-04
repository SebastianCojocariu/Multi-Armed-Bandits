from contextual_bandit import *

class ContextualEpsilonGreedy(ContextualBandit):
	def __init__(self, policy_args, model_args):
		super().__init__(policy_args=policy_args, model_args=model_args)
		self.p = policy_args["p"]
		self.decay = policy_args["decay"]
		self.policy_name = "ContextualEpsilonGreedy" + "(p: {}, decay: {}, oracle: {})".format(self.p, self.decay, model_args["class_model_name"])
		
		
	def compute_score(self, arm, context, normalising_factor):
		score = self.oracles[arm]["model"].predict(context)	
		score = self.augment_prediction(arm=arm, prediction=score, normalising_factor=normalising_factor)
		return score


	def update(self, arm, event):
		ContextualBandit.update(self, arm=arm, event=event)	
		self.p *= self.decay
		
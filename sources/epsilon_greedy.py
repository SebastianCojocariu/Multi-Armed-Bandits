from contextual_free_bandit import *

class EpsilonGreedy(ContextualFreeBandit):
	def __init__(self, policy_args):
		super().__init__(policy_args=policy_args)
		self.p = policy_args["p"]
		self.decay = policy_args["decay"]
		self.policy_name = "EpsilonGreedy" + "(p: {}, decay: {})".format(self.p, self.decay)
		

	def compute_score(self, arm):
		total_reward, n = self.states[arm]["total_reward"], self.states[arm]["n"]
		score = total_reward / n if n > 0 else float("Infinity")
		return score


	def update(self, arm, event):
		ContextualFreeBandit.update(self, arm=arm, event=event)
		self.p *= self.decay
		
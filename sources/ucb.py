from contextual_free_bandit import *

class UCB(ContextualFreeBandit):
	def __init__(self, policy_args):
		super().__init__(policy_args=policy_args)
		self.confidence_level = policy_args["confidence_level"]
		self.policy_name = "UCB" + "(conf_level: {}, p: {})".format(self.confidence_level, self.p)


	def compute_score(self, arm):
		total_reward, n = self.states[arm]["total_reward"], self.states[arm]["n"]
		score = float("Infinity") if n == 0 else total_reward / n +  np.sqrt(self.confidence_level * np.log(self.current_event_count) / n)
		return score
		
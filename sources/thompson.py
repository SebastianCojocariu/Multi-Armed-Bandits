from contextual_free_bandit import *

class Thompson(ContextualFreeBandit):
	def __init__(self, policy_args={}):
		super().__init__(policy_args=policy_args)
		self.policy_name = "Thompson" + "(p: {})".format(self.p)


	def compute_score(self, arm):
		alpha, beta = self.states[arm]["alpha"], self.states[arm]["beta"]
		score = np.random.beta(alpha, beta)
		return score


	def initialize_new_arms(self, event):
		for arm in event["available_arms"]:
			if arm not in self.states:
				self.states[arm] = {"total_reward": 0, "n": 0, "alpha": 1, "beta": 1}

				
	def update(self, arm, event):
		ContextualFreeBandit.update(self, arm=arm, event=event)
		self.states[arm]["alpha"] += event["reward"]
		self.states[arm]["beta"] += (1 - event["reward"])
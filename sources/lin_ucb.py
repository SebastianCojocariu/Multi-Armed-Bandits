from contextual_bandit import *

class LinUCB(ContextualBandit):
	def __init__(self, policy_args, model_args={}):
		super().__init__(policy_args=policy_args, model_args=model_args)
		self.alpha = policy_args["alpha"]
		self.policy_name = "LinUCB" + "(alpha: {}, p: {})".format(self.alpha, self.p)


	def compute_score(self, arm, context, normalising_factor):
		context = np.asarray(context).reshape(-1, 1)
		A_inverse, theta = self.oracles[arm]["A_inverse"], self.oracles[arm]["theta"]	
		score = np.transpose(theta).dot(context) + self.alpha * np.sqrt(np.transpose(context).dot(A_inverse).dot(context))
		return score


	def initialize_new_arms(self, event):
		context = np.asarray(event["context"])
		if len(context.shape) == 1:
			context = np.reshape(context, (1, -1))
		
		d = context.shape[-1]

		for arm in event["available_arms"]:
			if arm not in self.oracles:
				self.oracles[arm] = {"A": np.identity(d), "A_inverse": np.identity(d), "b": np.zeros((d, 1)), "theta": np.zeros((d, 1))}
			if arm not in self.states:
				self.states[arm] = {"total_reward": 0, "n": 0}


	def update(self, arm, event):
		self.total_rewards += event["reward"]
		self.current_event_count += 1

		self.states[arm]["total_reward"] += event["reward"]
		self.states[arm]["n"] += 1

		context = np.asarray(event["context"]).reshape(-1, 1)

		self.oracles[arm]["A"] += context.dot(np.transpose(context))
		self.oracles[arm]["b"] += event["reward"] * context

		self.oracles[arm]["A_inverse"] = np.linalg.inv(self.oracles[arm]["A"])
		self.oracles[arm]["theta"] = self.oracles[arm]["A_inverse"].dot(self.oracles[arm]["b"])

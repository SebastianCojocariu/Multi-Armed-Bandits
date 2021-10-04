from base_bandit import *

class ContextualFreeBandit(BaseBandit):
	def __init__(self, policy_args, model_args={}):
		super().__init__(policy_args=policy_args)


	def compute_score(self, arm):
		raise NotImplementedError()


	def choose_arm(self, event):
		available_arms = event["available_arms"]

		# Exploration
		if self.p > 0 and random.uniform(0, 1) < self.p:
			return random.choice(available_arms)
		
		# Exploitation
		max_prediction, best_arms = float("-Infinity"), []
		for arm in available_arms:
			score = self.compute_score(arm=arm)

			if score > max_prediction:
				max_prediction, best_arms = score, [arm]
			elif score == max_prediction:
				best_arms.append(arm)

		assert len(best_arms) > 0, "There should be at least 1 available arm. Check the logic again!"
		next_arm = random.choice(best_arms)

		return next_arm


	def initialize_new_arms(self, event):
		for arm in event["available_arms"]:
			if arm not in self.states:
				self.states[arm] = {"total_reward": 0, "n": 0}


	def update(self, arm, event):
		self.total_rewards += event["reward"]
		self.current_event_count += 1
		self.states[arm]["total_reward"] += event["reward"]
		self.states[arm]["n"] += 1

	
	def save_event(self, arm, event):
		pass # <--- No need to save any event, as we won't be using them.

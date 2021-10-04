from contextual_bandit import *

class ContextualAdaptiveGreedy(ContextualBandit):
	def __init__(self, policy_args, model_args):
		super().__init__(policy_args=policy_args, model_args=model_args)
		self.threshold = policy_args["threshold"]
		self.decay = policy_args["decay"]
		self.policy_name = "ContextualAdaptiveGreedy" + "(threshold: {}, decay: {}, p: {}, oracle: {})".format(self.threshold, self.decay, self.p, model_args["class_model_name"])


	def compute_score(self, arm, context, normalising_factor):
		score = self.oracles[arm]["model"].predict(context)	
		score = self.augment_prediction(arm=arm, prediction=score, normalising_factor=normalising_factor)
		return score


	def choose_arm(self, event):
		context = event["context"]
		available_arms = event["available_arms"]

		if self.p > 0 and random.uniform(0, 1) < self.p:
			return random.choice(available_arms)

		normalising_factor = self.compute_normalizing_factor(available_arms=available_arms)

		max_prediction, best_arms = float("-Infinity"), []
		for arm in available_arms:
			score = self.compute_score(arm=arm, context=context, normalising_factor=normalising_factor)

			if score > max_prediction:
				max_prediction, best_arms = score, [arm]
			elif score == max_prediction:
				best_arms.append(arm)

		assert len(best_arms) > 0, "There should be at least 1 available arm. Check the logic again!"

		if max_prediction > self.threshold:
			next_arm = random.choice(best_arms)
		else:
			next_arm = random.choice(available_arms)

		return next_arm


	def update(self, arm, event):
		ContextualBandit.update(self, arm=arm, event=event)
		self.threshold *= self.decay
		
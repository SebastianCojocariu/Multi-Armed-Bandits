from contextual_bandit import *

class BootstrappedThompson(ContextualBandit):
	def __init__(self, policy_args, model_args):
		super().__init__(policy_args=policy_args, model_args=model_args)
		self.m = policy_args["m"]
		self.policy_name = "BootstrappedThompson" + "(m: {}, p: {}, oracle: {})".format(self.m, self.p, model_args["class_model_name"])


	def compute_score(self, arm, context, normalising_factor):
		s = random.randint(0, self.m - 1)
		score = self.oracles[arm]["models"][s].predict(context)
		score = self.augment_prediction(arm=arm, prediction=score, normalising_factor=normalising_factor)
		return score


	def initialize_new_arms(self, event):
		for arm in event["available_arms"]:
			if arm not in self.oracles:
				self.oracles[arm] = {"models": [Oracle(args=self.model_args)] * self.m, "last_pool_sizes": [0] * self.m} 
			if arm not in self.states:
				self.states[arm] = {"total_reward": 0, "n": 0}


	def update(self, arm, event):
		self.total_rewards += event["reward"]
		self.current_event_count += 1

		self.states[arm]["total_reward"] += event["reward"]
		self.states[arm]["n"] += 1
		
		threshold_retrain_percent = self.policy_args["threshold_retrain_percent"] if "threshold_retrain_percent" in self.policy_args else 0.1

		for i in range(self.m):
			curr_history_size = len(self.history[arm]["X"])
			last_train_size = self.oracles[arm]["last_pool_sizes"][i]

			if last_train_size == 0 or (curr_history_size - last_train_size) / last_train_size >= threshold_retrain_percent: 
				(X, y) = self.__bootstrapped_resample(arm=arm)
				new_oracle = Oracle(args=self.model_args)
				fitted = new_oracle.fit(X=X, y=y)

				if fitted:
					self.oracles[arm]["models"][i] = new_oracle
					self.oracles[arm]["last_pool_sizes"][i] = len(X)


	def __bootstrapped_resample(self, arm):
		zipped = list(zip(self.history[arm]["X"], self.history[arm]["y"]))
		zipped = random.choices(zipped, k=len(zipped))

		X = np.asarray([x for (x, _) in zipped])
		y = np.asarray([y for (_, y) in zipped])

		return (X, y)
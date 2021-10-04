from base_bandit import *

class ContextualBandit(BaseBandit):
	def __init__(self, policy_args, model_args):
		super().__init__(policy_args=policy_args)
		self.model_args = model_args # <--- arguments to instantiate the oracle(s)
		self.oracles = {} # <--- to store the actual oracles per arms
		self.history = {} # <--- to store the processed events (or the relevant parts of them)
		self.states_heuristics = policy_args["states_heuristics"] if "states_heuristics" in policy_args else False # <--- whether to apply heuristics or not


	def compute_score(self, arm, context, normalising_factor):
		raise NotImplementedError()


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
		next_arm = random.choice(best_arms)

		return next_arm


	def compute_normalizing_factor(self, available_arms):
		normalising_factor = 0
		for arm in available_arms:
			total_rewards, n = self.states[arm]["total_reward"], self.states[arm]["n"]
			if n == 0:
				normalising_factor = float("Infinity")
				break
			else:
				normalising_factor += total_rewards / n
		
		return normalising_factor


	def augment_prediction(self, arm, prediction, normalising_factor=1.0):
		final_prediction = prediction

		if self.states_heuristics:
			expected_reward = self.get_estimated_reward(arm=arm)
			
			if normalising_factor == 0:
				expected_reward = 1
			elif normalising_factor == float("Infinity"):
				expected_reward = 0 if expected_reward != float("Infinity") else float("Infinity")
			else:
				expected_reward = expected_reward / normalising_factor

			final_prediction += expected_reward
		
		return final_prediction


	def initialize_new_arms(self, event):
		for arm in event["available_arms"]:
			if arm not in self.oracles:
				self.oracles[arm] = {"model": Oracle(args=self.model_args), "last_pool_size": 0}
			if arm not in self.states:
				self.states[arm] = {"total_reward": 0, "n": 0}

	
	def update(self, arm, event):
		self.total_rewards += event["reward"]
		self.current_event_count += 1

		self.states[arm]["total_reward"] += event["reward"]
		self.states[arm]["n"] += 1

		threshold_retrain_percent = self.policy_args["threshold_retrain_percent"] if "threshold_retrain_percent" in self.policy_args else 0.1
		
		curr_history_size = len(self.history[arm]["X"])
		last_train_size = self.oracles[arm]["last_pool_size"] 

		# update the oracle only if the number of features from the last fit increased by at least threshold_retrain_percent
		if last_train_size == 0 or (curr_history_size - last_train_size) / last_train_size >= threshold_retrain_percent:
			new_oracle = Oracle(args=self.model_args)
			fitted = new_oracle.fit(X=self.history[arm]["X"], y=self.history[arm]["y"])

			if fitted:
				self.oracles[arm] = {"model": new_oracle, "last_pool_size": curr_history_size}

		
	def save_event(self, arm, event):
		context, reward = event["context"], event["reward"]
		if arm not in self.history:
			self.history[arm] = {"X": [context], "y": [reward]}
		else:
			self.history[arm]["X"].append(context)
			self.history[arm]["y"].append(reward)

import numpy as np
import random
import sys
import random
import time
from oracle import Oracle
sys.path.append(".")

class BaseBandit():
	def __init__(self, policy_args={}):
		self.policy_args = policy_args # <--- policy arguments
		self.p = policy_args["p"] if "p" in policy_args else 0.0 # <--- random parameter used for exploration
		self.current_event_count = 0 # <--- total events processed so far
		self.total_rewards = 0 # <--- total rewards accumulated so far
		self.policy_name = "" # <--- should be updated by each policy inheriting from BaseBandit
		self.states = {} # <--- to store the internal states of the policy
		self.ctr = 0 # <--- to store the current CTR
	

	def get_estimated_reward(self, arm):
		if arm not in self.states:
			return float("Infinity") # <---- give a high priority to unexplored arms

		total_reward, n = self.states[arm]["total_reward"], self.states[arm]["n"]
		score = float("Infinity") if n == 0 else total_reward / n

		return score


	def choose_arm(self, event):
		raise NotImplementedError()


	def initialize_new_arms(self, event):
		raise NotImplementedError()


	def update(self, arm, event):
		raise NotImplementedError()


	def save_event(self, arm, event):
		raise NotImplementedError()


	def step(self, event):
		target_arm = event["target_arm"]
		available_arms = event["available_arms"]
		reward = None

		# update the policy with new arms (from available arms from event).
		self.initialize_new_arms(event=event)
		
		# select the next arm based on the policy heuristics
		next_arm = self.choose_arm(event=event)

		# Offline policy rule. Update only if the target_arm is the same as the one predicted by the policy
		if next_arm == target_arm:
			reward = event["reward"]

			# save the current event (mainly used for oracle update in ContextualBandit)
			self.save_event(arm=next_arm, event=event)

			# update the internal states. Should update self.total_rewards and self.current_event_count too
			self.update(arm=next_arm, event=event)

			# update the current CTR
			self.ctr = 0 if self.current_event_count == 0 else self.total_rewards / self.current_event_count			
		
		# reward = None means that the current event was not used for update
		return reward, self.ctr


	def fit(self, event_generator, debug=False):
		ctr_list, current_event_idx, last_introduced_event_idx = [], 0, 0
		start_time = time.time()
		print("#### Started fitting: {} ####\n".format(self.policy_name))
		
		while(True):
			try:
				event = next(event_generator)
			except StopIteration:
				break

			current_event_idx += 1
			reward, ctr = self.step(event=event)

			# if an update occured
			if reward is not None:
				last_introduced_event_idx += 1
				ctr_list.append(ctr)
				
				if debug and last_introduced_event_idx % 1000 == 0:
					end_time = time.time()
					print("[Time: {}]{} / {} => CTR: {}".format(end_time - start_time, current_event_idx, last_introduced_event_idx, ctr))
					start_time = end_time
		
		print("#### Finished fitting: {} ####\n".format(self.policy_name))
		return {"total_rewards": np.asarray(ctr_list), "policy_name": self.policy_name}

		
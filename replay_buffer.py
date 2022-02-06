import copy
import time

import numpy as np
import ray
import tensorflow as tf

import my_config
import network
import self_play


@ray.remote
class ReplayBuffer:
	"""
	Class which run in a dedicated thread to store played games and generate batch.
	"""

	def __init__(self, initial_checkpoint, initial_buffer, config:my_config.Config):
		self.config = config
		self.buffer = copy.deepcopy(initial_buffer)
		self.num_played_games = initial_checkpoint["num_played_games"]
		self.num_played_steps = initial_checkpoint["num_played_steps"]
		self.first_game_id=self.num_played_games - len(self.buffer)
		self.total_samples = sum(
			[game_history.length for game_history in self.buffer.values()]
		)
		if self.total_samples != 0:
			print(f'Replay buffer initialized with {self.total_samples} samples ({self.num_played_games} games).\n')

		# Fix random generator seed
		np.random.seed(self.config.seed)

	def save_game(self, game_history, shared_storage=None):
		if self.config.PER:
			if game_history.priorities is not None:
				# Avoid read only array when loading replay buffer from disk
				game_history.priorities = np.copy(game_history.priorities)
			else:
				# Initial priorities for the prioritized replay (See paper appendix Training)
				priorities = []
				for i, root_value in enumerate(game_history.root_values):
					priority = (
						np.abs(
							root_value - self.compute_target_value(game_history, i)
						)
						** self.config.PER_alpha
					)
					priorities.append(priority)

				game_history.priorities = np.array(priorities, dtype="float32")
				game_history.game_priority = np.max(game_history.priorities)

		self.buffer[self.num_played_games] = game_history
		self.num_played_games += 1
		self.num_played_steps += game_history.length
		self.total_samples += game_history.length

		if self.config.replay_buffer_size < len(self.buffer):
			del_id = self.num_played_games - len(self.buffer)
			self.total_samples -= self.buffer[del_id].length
			del self.buffer[del_id]

		if shared_storage:
			shared_storage.set_info.remote("num_played_games", self.num_played_games)
			shared_storage.set_info.remote("num_played_steps", self.num_played_steps)
		if self.config.save_game_to_file:
			game_history.save(f'saved_games/{self.num_played_games}.record')
	def load_games(self, first_game_id, length):
		for i in range(first_game_id, first_game_id+length):
			game_history=self_play.GameHistory()
			game_history.load(f'saved_games/{i}.record')#not implemented yet
			self.buffer[i]=game_history
			self.num_played_games += 1
			self.num_played_steps += game_history.length
			self.total_samples += game_history.length
	def get_buffer(self):
		return self.buffer

	def get_batch(self):
		#samples always start from state followed by actions of type 0 (move)
		(
			index_batch,
			observation_batch,
			action_batch,
			reward_batch,
			value_batch,
			policy_batch,
		) = ([], [], [], [], [], [], [])
		weight_batch = [] if self.config.PER else None

		for game_id, game_history, game_prob in self.sample_n_games(self.config.batch_size):
			game_pos, pos_prob = self.sample_position(game_history)

			values, rewards, policies, actions = self.make_target(
				game_history, game_pos
			)

			index_batch.append([game_id, game_pos])
			observation_batch.append(
				game_history.get_observation(game_pos)
			)
			action_batch.append(actions)
			value_batch.append(values)
			reward_batch.append(rewards)
			policy_batch.append(policies)
			if self.config.PER:
				weight_batch.append(1 / (self.total_samples * game_prob * pos_prob))

		if self.config.PER:
			weight_batch = np.array(weight_batch, dtype="float32") / max(
				weight_batch
			)

		# observation_batch: batch, channels, height, width
		# action_batch: batch, num_unroll_steps+1
		# value_batch: batch, num_unroll_steps+1
		# reward_batch: batch, num_unroll_steps+1
		# policy_batch: batch, num_unroll_steps+1, len(action_space(type 0))
		# weight_batch: batch
		# gradient_scale_batch: batch, num_unroll_steps+1
		return (
			index_batch,
			(
				observation_batch,
				action_batch,
				value_batch,
				reward_batch,
				policy_batch,
				weight_batch,
			),
		)

	def sample_game(self, force_uniform=False):
		"""
		Sample game from buffer either uniformly or according to some priority.
		See paper appendix Training.
		"""
		game_prob = None
		if self.config.PER and not force_uniform:
			game_probs = np.array(
				[game_history.game_priority for game_history in self.buffer.values()],
				dtype="float32",
			)
			game_probs /= np.sum(game_probs)
			game_index = np.random.choice(len(self.buffer), p=game_probs)
			game_prob = game_probs[game_index]
		else:
			game_index = np.random.choice(len(self.buffer))
		game_id = self.num_played_games - len(self.buffer) + game_index

		return game_id, self.buffer[game_id], game_prob

	def sample_n_games(self, n_games, force_uniform=False):
		if self.config.PER and not force_uniform:
			game_id_list = []
			game_probs = []
			for game_id, game_history in self.buffer.items():
				game_id_list.append(game_id)
				game_probs.append(game_history.game_priority)
			game_probs = np.array(game_probs, dtype="float32")
			game_probs /= np.sum(game_probs)
			game_prob_dict = dict([(game_id, prob) for game_id, prob in zip(game_id_list, game_probs)])
			selected_games = np.random.choice(game_id_list, n_games, p=game_probs)
		else:
			selected_games = np.random.choice(list(self.buffer.keys()), n_games)
			game_prob_dict = {}
		ret = [(game_id, self.buffer[game_id], game_prob_dict.get(game_id))
			   for game_id in selected_games]
		return ret

	def sample_position(self, game_history, force_uniform=False):
		"""
		Sample position from game either uniformly or according to some priority.
		See paper appendix Training.
		"""
		position_prob = None
		if self.config.PER and not force_uniform:
			position_probs = game_history.priorities / sum(game_history.priorities)
			position_index = np.random.choice(game_history.length, p=position_probs)
			position_prob = position_probs[position_index]
		else:
			position_index = np.random.choice(game_history.length)
		while position_index+1<game_history.length and game_history.type_history[position_index+1]!=0:
			position_index+=1
		if position_index+1==game_history.length:#this could never happen because the last action must be type 0
			print('something weird happened... see replay_buffer.ReplayBuffer.sample_position')
			return self.sample_position(game_history, force_uniform)
		return position_index, position_prob

	def update_game_history(self, game_id, game_history):
		# The element could have been removed since its selection and update
		if next(iter(self.buffer)) <= game_id:
			if self.config.PER:
				# Avoid read only array when loading replay buffer from disk
				game_history.priorities = np.copy(game_history.priorities)
			self.buffer[game_id] = game_history

	def update_priorities(self, priorities, index_info):
		"""
		Update game and position priorities with priorities calculated during the training.
		See Distributed Prioritized Experience Replay https://arxiv.org/abs/1803.00933
		"""
		for i in range(len(index_info)):
			game_id, game_pos = index_info[i]

			# The element could have been removed since its selection and training
			if next(iter(self.buffer)) <= game_id:
				# Update position priorities
				priority = priorities[i, :]
				start_index = game_pos
				end_index = min(
					game_pos + len(priority), len(self.buffer[game_id].priorities)
				)
				self.buffer[game_id].priorities[start_index:end_index] = priority[
					: end_index - start_index
				]

				# Update game priorities
				self.buffer[game_id].game_priority = np.max(
					self.buffer[game_id].priorities
				)

	def compute_target_value(self, game_history, index):
		# The value target is the discounted root value of the search tree td_steps into the
		# future, plus the discounted sum of all rewards until then.
		bootstrap_index = index + self.config.td_steps
		if bootstrap_index < len(game_history.root_values):
			root_values = (
				game_history.root_values
				if game_history.reanalysed_predicted_root_values is None
				else game_history.reanalysed_predicted_root_values
			)
			last_step_value = root_values[bootstrap_index]

			value = last_step_value * self.config.discount ** self.config.td_steps
		else:
			value = 0
		dis=1.
		for reward in game_history.reward_history[index + 1 : min(game_history.length,bootstrap_index + 1)]:
			# The value is oriented from the perspective of the current player
			value += reward * dis
			dis*=self.config.discount

		return value
	def make_target(self, game_history, state_index):
		"""
		Generate targets for every unroll steps.
		only generate for actions with type 0
		for type 1, all target are None, and will not count to loss in training.
		#another idea: value: the same, reward: 0, policy: uniform
		#this may improve quality of dynamics model
		"""
		target_values, target_rewards, target_policies, actions = [], [], [], []
		def append_none(action):
			actions.append(action)
		for current_index in range(
			state_index, state_index + self.config.num_unroll_steps + 1
		): 
			if game_history.type_history[current_index]==1:
				append_none(game_history.action_history[current_index])
				continue
			value = self.compute_target_value(game_history, current_index)

			if current_index < len(game_history.root_values):
				target_values.append(value)
				target_rewards.append(game_history.reward_history[current_index])
				target_policies.append(game_history.child_visits[current_index])
				actions.append(game_history.action_history[current_index])#actions[0] is not used, so trainer.update_weights start recurrent inference from actions[1]
			elif current_index == len(game_history.root_values):
				target_values.append(0)
				target_rewards.append(game_history.reward_history[current_index])
				# Uniform policy
				target_policies.append(
					[
						1 / len(self.config.action_space_type0)
						for _ in self.config.action_space_type0
					]
				)
				actions.append(game_history.action_history[current_index])
			else:
				# States past the end of games are treated as absorbing states
				target_values.append(0)
				target_rewards.append(0)
				# Uniform policy
				target_policies.append(
					[
						1 / len(game_history.child_visits[0])
						for _ in range(len(game_history.child_visits[0]))
					]
				)
				actions.append(np.random.choice(self.config.action_space_type0))

		return target_values, target_rewards, target_policies, actions


@ray.remote
class Reanalyse:
	"""
	Class which run in a dedicated thread to update the replay buffer with fresh information.
	See paper appendix Reanalyse.
	"""

	def __init__(self, initial_checkpoint, config):
		self.config = config

		# Fix random generator seed
		np.random.seed(self.config.seed)
		tf.random.set_seed(self.config.seed)

		# Initialize the network
		self.model = network.Network(self.config)
		self.model.set_weights(initial_checkpoint["weights"])

		self.num_reanalysed_games = initial_checkpoint["num_reanalysed_games"]

	def reanalyse(self, replay_buffer, shared_storage):
		while ray.get(shared_storage.get_info.remote("num_played_games")) < 1:
			time.sleep(0.1)

		while ray.get(
			shared_storage.get_info.remote("training_step")
		) < self.config.training_steps and not ray.get(
			shared_storage.get_info.remote("terminate")
		):
			self.model.set_weights(ray.get(shared_storage.get_info.remote("weights")))

			game_id, game_history, _ = ray.get(
				replay_buffer.sample_game.remote(force_uniform=True)
			)

			# Use the last model to provide a fresher, stable n-step value (See paper appendix Reanalyze)
			if self.config.reanalyse:
				observations = [
					game_history.get_stacked_observations(
						i, self.config.stacked_observations
					)
					for i in range(len(game_history.root_values))
				]
				observations=np.array(observations,dtype=np.float32)
				values = network.support_to_scalar(
					self.model.initial_inference(observations)['value'],
					self.config.support_size,
				)
				game_history.reanalysed_predicted_root_values =	tf.squeeze(values).numpy()

			replay_buffer.update_game_history.remote(game_id, game_history)
			self.num_reanalysed_games += 1
			shared_storage.set_info.remote(
				"num_reanalysed_games", self.num_reanalysed_games
			)

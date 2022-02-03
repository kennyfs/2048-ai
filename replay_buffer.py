from copy import deepcopy as dc
import time

import numpy
import ray
import tensorflow as tf

import network

@ray.remote
class ReplayBuffer:
	"""
	Class which run in a dedicated thread to store played games and generate batch.
	"""

	def __init__(self, initial_checkpoint, initial_buffer, config):
		self.config = config
		self.buffer = dc(initial_buffer)
		self.num_played_games = initial_checkpoint["num_played_games"]
		self.num_played_steps = initial_checkpoint["num_played_steps"]
		self.total_samples = sum(
			[game_history.length for game_history in self.buffer.values()]
		)
		if self.total_samples != 0:
			print(f'Replay buffer initialized with {self.total_samples} samples ({self.num_played_games} games).\n')

		# Fix random generator seed
		numpy.random.seed(self.config.seed)

	def save_game(self, game_history, shared_storage=None):

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

	def get_buffer(self):
		return self.buffer

	def get_batch(self):
		(
			index_batch,
			observation_batch,
			action_batch,
			reward_batch,
			value_batch,
			policy_batch,
			gradient_scale_batch,
		) = ([], [], [], [], [], [], [])

		for game_id, game_history in self.sample_n_games(self.config.batch_size):
			game_pos = self.sample_position(game_history)

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


		# observation_batch: batch, channels, height, width
		# action_batch: batch, num_unroll_steps+1
		# value_batch: batch, num_unroll_steps+1
		# reward_batch: batch, num_unroll_steps+1
		# policy_batch: batch, num_unroll_steps+1, len(action_space(type 0))
		# gradient_scale_batch: batch, num_unroll_steps+1
		return (
			index_batch,
			(
				observation_batch,
				action_batch,
				value_batch,
				reward_batch,
				policy_batch,
			),
		)

	def sample_game(self):
		"""
		Sample game from buffer either uniformly or according to some priority.
		See paper appendix Training.
		"""
		game_index = numpy.random.choice(len(self.buffer))
		game_id = self.num_played_games - len(self.buffer) + game_index

		return game_id, self.buffer[game_id]

	def sample_n_games(self, n_games):
		selected_games = numpy.random.choice(list(self.buffer.keys()), n_games)
		ret = [(game_id, self.buffer[game_id])
			   for game_id in selected_games]
		return ret

	def sample_position(self, game_history):
		"""
		Sample position from game either uniformly or according to some priority.
		See paper appendix Training.
		"""
		position_index = numpy.random.choice(game_history.length)

		return position_index

	def update_game_history(self, game_id, game_history):
		# The element could have been removed since its selection and update
		if next(iter(self.buffer)) <= game_id:
			self.buffer[game_id] = game_history

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
			target_values.append(None)
			target_rewards.append(None)
			target_policies.append(None)
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
				actions.append(game_history.action_history[current_index])
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
				pass
				#don't append anything
				'''
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
				actions.append(numpy.random.choice(self.config.action_space))
				'''

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
		numpy.random.seed(self.config.seed)
		tf.random.set_seed(self.config.seed)

		# Initialize the network
		self.model = network.Network(self.config)
		self.model.set_weights(initial_checkpoint["weights"])
		self.model.eval()

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
				replay_buffer.sample_game.remote()
			)

			# Use the last model to provide a fresher, stable n-step value (See paper appendix Reanalyze)
			if self.config.use_last_model_value:
				observations = [
					game_history.get_stacked_observations(
						i, self.config.stacked_observations
					)
					for i in range(len(game_history.root_values))
				]
				results=self.model.initial_inference(observations)
				if self.config.support:
					values = network.support_to_scalar(
						results,
						self.config.support_size,
					)
				else:
					values=results
				game_history.reanalysed_predicted_root_values =	tf.squeeze(values).numpy()

			replay_buffer.update_game_history.remote(game_id, game_history)
			self.num_reanalysed_games += 1
			shared_storage.set_info.remote(
				"num_reanalysed_games", self.num_reanalysed_games
			)

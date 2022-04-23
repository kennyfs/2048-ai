import copy
import random
import time

import numpy as np
import tensorflow as tf

import my_config
import network


def scale_gradient(tensor, scale):
	"""Scales the gradient for the backward pass."""
	return tensor * scale + tf.stop_gradient(tensor) * (1 - scale)
timers = [0]*10
class Trainer:
	"""
	Class which run in a dedicated thread to train a neural network and save it
	in the shared storage.
	"""

	def __init__(self, initial_checkpoint, model:network.Network, config:my_config.Config):
		self.config = config
		self.model = model
		'''
		# Initialize the network
		self.model = network.Network(self.config)
		self.model.set_weights(copy.deepcopy(initial_checkpoint["weights"]))
		'''
		self.training_step = initial_checkpoint["training_step"]
		self.l2_weight = config.l2_weight

		self.batch_size = self.config.batch_size
		self.num_unroll_steps = self.config.num_unroll_steps
		# Initialize the optimizer
		if self.config.optimizer == "SGD":
			self.optimizer = tf.keras.optimizers.SGD(
				learning_rate = self.config.learning_rate_init,
				momentum = self.config.momentum,
			)
		elif self.config.optimizer == "Adam":#the best
			self.optimizer = tf.keras.optimizers.Adam(
				learning_rate = self.config.learning_rate_init,
			)
		elif self.config.optimizer == "Adadelta":
			self.optimizer = tf.keras.optimizers.Adadelta(
				learning_rate = self.config.learning_rate_init,
			)
		else:
			raise NotImplementedError(
				f"{self.config.optimizer} is not implemented. You can set up the optimizer manually in trainer.py."
			)

	def run_update_weights(self, replay_buffer, shared_storage, max_steps = None, force_training:bool = False):#max_steps is for observing loss
		# Wait for the replay buffer to be filled
		assert shared_storage.get_info("num_played_games") > 0, 'no enough games, get 0'
		self.training_step = shared_storage.get_info('training_step')
		# Training loop
		shared_storage.clear_loss()
		shared_storage.clear_output()
		start_step = self.training_step
		while (self.training_step / max(
				1, shared_storage.get_info("num_played_steps"))
				 < self.config.training_steps_to_selfplay_steps_ratio
				and
				self.training_step < self.config.training_steps
				or
				force_training):
			st = time.time()
			index_batch, batch = replay_buffer.get_batch(self.batch_size)
			print(f'generating data consumed {time.time()-st} seconds.')
			if self.config.optimizer != 'Adadelta':
				self.update_learning_rate()
			print(f'training_step:{self.training_step}, lr = {self.optimizer.learning_rate}')
			st = time.time()
			(
				priorities,
				total_loss,
				value_loss,
				reward_loss,
				chance_loss,
				policy_loss,
				l2_loss,

				value_initial,
				value_recurrent,
				reward,
				value_initial_delta,
				value_recurrent_delta,
				reward_delta,
			) = self.update_weights(batch, shared_storage)
			print([int(t/sum(timers)*100) for t in timers])
			print(f'training consumed {time.time()-st} seconds.')

			if self.config.PER:
				# Save new priorities in the replay buffer (See https://arxiv.org/abs/1803.00933)
				replay_buffer.update_priorities(priorities, index_batch)

			# Save to the shared storage
			shared_storage.set_info(
				{
					"training_step": self.training_step,
					"learning_rate": self.optimizer.learning_rate,
				}
			)
			shared_storage.append_loss(total_loss, value_loss, reward_loss, chance_loss, policy_loss, l2_loss)
			shared_storage.append_output(value_initial, value_recurrent, reward, value_initial_delta, value_recurrent_delta, reward_delta)
			# Managing the self-play / training ratio
			print(f'ratio:{self.training_step / max(1, shared_storage.get_info("num_played_steps"))}')
			if max_steps != None and self.training_step-start_step >= max_steps:
				break
		#shared_storage.save_weights(copy.deepcopy(self.model.get_weights()))
		#shared_storage.save()
		# loss log:
		# clear losses saved in shared storage when training start.
		# train until training to selfplay steps ratio is up to the config.

		# log has a long series of losses.
	def update_weights(self, batch, shared_storage):
		"""
		Perform one training step.update_learning_rate
		"""
		(
			observation_batch,
			target_chance_batch,
			action_batch,
			target_value_batch,
			target_reward_batch,
			target_policy_batch,
			weight_batch,
		) = batch#batches are all np.array
		#for debug, check the shape of the batch, maybe is useful
		#for b in batch:
		#	print(b.shape)
		if random.random() < 0.1:
			#It's important to check data sometimes, because the data is not always good.
			print('training')
			print(f'value:{target_value_batch[0, :]}')
			print(f'reward:{target_reward_batch[0, :]}')
			print(f'policy:{target_policy_batch[0, :]}')
		# Keep values as scalars for calculating the priorities for the prioritized replay
		target_value_scalar = np.copy(target_value_batch)
		target_reward_scalar = np.copy(target_reward_batch)
		if self.config.PER:
			priorities = np.zeros_like(target_value_scalar)
		else:
			priorities = None

		# observation_batch: batch, channels, height, width
		# target_chance_batch: batch, num_unroll_steps(no +1), 2 or 1, board size, board size
		# action_batch: batch, num_unroll_steps # UDLR
		# target_value_batch: batch, num_unroll_steps+1 #+1 for the those of initial reference
		# target_reward_batch: batch, num_unroll_steps+1
		# target_policy_batch: batch, num_unroll_steps+1, action_space_size
		if self.config.support:
			target_value_batch = network.scalar_to_support(target_value_batch, self.config.support)
			target_reward_batch = network.scalar_to_support(target_reward_batch, self.config.support)
		else:
			target_value_batch = tf.expand_dims(target_value_batch, axis = -1)
			target_reward_batch = tf.expand_dims(target_reward_batch, axis = -1)
		target_chance_batch = tf.reshape(target_chance_batch, (self.batch_size, self.num_unroll_steps, -1))#flatten for entropy loss
		# target_value: batch, num_unroll_steps+1, 2*support+1
		# target_reward: batch, num_unroll_steps+1, 2*support+1
		# target_chance: batch, (2*) board size**2

		# if not using support
		# target_value: batch, num_unroll_steps+1
		# target_reward: batch, num_unroll_steps+1
		assert (
			list(observation_batch.shape) == [self.batch_size]+self.config.observation_shape and
			list(target_chance_batch.shape) == [self.batch_size, self.num_unroll_steps, self.config.chance_output*self.config.board_size**2] and
			list(action_batch.shape) == [self.batch_size, self.num_unroll_steps, 2] and
			list(target_value_batch.shape) == [self.batch_size, self.num_unroll_steps+1, 2*self.config.support+1] and
			list(target_reward_batch.shape) == [self.batch_size, self.num_unroll_steps+1, 2*self.config.support+1] and
			list(target_policy_batch.shape) == [self.batch_size, self.num_unroll_steps+1, 4] and
			(not self.config.PER or list(weight_batch.shape) == [self.batch_size])), f'batch shape error, {observation_batch.shape}, {action_batch.shape}, {target_value_batch.shape}, {target_reward_batch.shape}, {target_policy_batch.shape}, {weight_batch.shape}'
		class tmp:#I don't know better solution
			def __init__(self, value = None):
				self.value = value
			def set(self, value):
				self.value = value
		step = tmp(0)
		last_loss = tmp()
		last_value_loss = tmp()
		last_reward_loss = tmp()
		last_chance_loss = tmp()
		last_policy_loss = tmp()
		last_l2_loss = tmp()

		last_value_initial = tmp()
		last_value_recurrent = tmp()
		last_value_initial_delta = tmp()
		last_value_recurrent_delta = tmp()
		last_reward = tmp()
		last_reward_delta = tmp()
		method = 2
		if method == 1:
			new_action_batch = tf.zeros((self.num_unroll_steps, self.batch_size, 4, self.config.board_size, self.config.board_size))
			new_random_action_batch = tf.zeros((self.num_unroll_steps, self.batch_size, 2, self.config.board_size, self.config.board_size))
			for i in range(self.num_unroll_steps):
				new_action_batch = tf.tensor_scatter_nd_update(new_action_batch, [i], network.action_to_onehot(action_batch[:, i, 0], self.config.board_size))
				new_random_action_batch = tf.tensor_scatter_nd_update(new_random_action_batch, [i], network.random_action_to_onehot(action_batch[:, i, 1], self.config.board_size))
		elif method == 2:
			action_batch = tf.reshape(action_batch, (self.num_unroll_steps*self.batch_size, 2))
			new_action_batch = network.action_to_onehot(action_batch[:, 0], self.config.board_size)
			new_random_action_batch = network.random_action_to_onehot(action_batch[:, 1], self.config.board_size)
			new_action_batch = tf.reshape(new_action_batch, (self.num_unroll_steps, self.batch_size, 4, self.config.board_size, self.config.board_size))
			new_random_action_batch = tf.reshape(new_random_action_batch, (self.num_unroll_steps, self.batch_size, 2, self.config.board_size, self.config.board_size))
		else:
			raise NotImplementedError(f'method {method}not implemented')
		action_batch = new_action_batch
		random_action_batch = new_random_action_batch
		## Generate predictions
		def loss_fn():
			st=time.time()
			output = self.model.initial_inference(
				observation_batch
			)#defined as time 0
			hidden_state = output.hidden_state
			value = output.value
			reward = tf.zeros_like(value)
			policy_logits = output.policy
			predictions = [(value, reward, policy_logits)]
			chances = []
			timers[0] += time.time()-st
			for i in range(self.num_unroll_steps):### start to check data processing here
				st=time.time()
				chance = self.model.chance([hidden_state, action_batch[i, :, :, :, :]])
				timers[1] += time.time()-st
				st=time.time()
				chances.append(chance)
				output = self.model.recurrent_inference(
					hidden_state, action_batch[i, :, :, :, :], random_action_batch[i, :, :, :, :], onehotted = True
				)
				timers[2] += time.time()-st
				st=time.time()
				reward = output.reward
				hidden_state = output.hidden_state
				hidden_state = scale_gradient(hidden_state, 0.5)
				value = output.value
				policy_logits = output.policy
				# Scale the gradient at the start of the dynamics function (See paper appendix Training)
				hidden_state = scale_gradient(hidden_state, 0.5)
				predictions.append((value, reward, policy_logits))
				timers[3] += time.time()-st
			assert len(predictions) == self.num_unroll_steps+1, f'len(predictions):{len(predictions)}\nself.num_unroll_steps:{self.num_unroll_steps}'
			#maybe there should be more assertion
			
			#predictions[t] = output at time t*2
			# predictions: num_unroll_steps+1, 3(batch, (2*support+1 | 2*support+1 | 9)) (according to the 2nd dim)
			# chances : num_unroll_steps, batch, chance_output*board_size**2
			total_value_loss, total_reward_loss, total_chance_loss, total_policy_loss = [tf.zeros((self.batch_size)) for _ in range(4)]
			# shape of losses: batch_size
			st=time.time()
			for i, prediction in enumerate(predictions):
				value, reward, policy_logits = prediction
				target_value = target_value_batch[:, i, :]
				target_reward = target_reward_batch[:, i, :]
				target_policy = target_policy_batch[:, i, :]
				value_loss, reward_loss, policy_loss = self.loss_function(value, reward, policy_logits, target_value, target_reward, target_policy, self.config.support != 0)
				if i == 0:
					total_value_loss += value_loss
					total_policy_loss += policy_loss
				else:
					total_value_loss += scale_gradient(value_loss, 1.0/(self.num_unroll_steps-1))
					total_reward_loss += scale_gradient(reward_loss, 1.0/(self.num_unroll_steps-1))
					total_policy_loss += scale_gradient(policy_loss, 1.0/(self.num_unroll_steps-1))
				if self.config.support:
					pred_value_scalar = network.support_to_scalar(value, self.config.support, True)
				else:
					pred_value_scalar = tf.reshape(value, (-1))
				if step.value == 0:
					if i == 0:
						index = random.randint(0, self.batch_size-1)
						last_value_initial_delta.set(pred_value_scalar[index]-target_value_scalar[index][i])
						last_value_initial.set(pred_value_scalar[index])
					elif i == 1:
						index = random.randint(0, self.batch_size-1)
						last_value_recurrent_delta.set(pred_value_scalar[index]-target_value_scalar[index][i])
						last_value_recurrent.set(pred_value_scalar[index])
						index = random.randint(0, self.batch_size-1)
						reward = tf.expand_dims(reward[index], 0)
						last_reward_delta.set(network.support_to_scalar(reward, self.config.support, True)[0].numpy()-target_reward_scalar[index][i])
						last_reward.set(network.support_to_scalar(reward, self.config.support, True)[0].numpy())
				if self.config.PER:
					priorities[:, i] = (
						np.abs(pred_value_scalar - target_value_scalar[:, i])
						** self.config.PER_alpha
					)
			timers[4] += time.time()-st
			st=time.time()
			for i, chance in enumerate(chances):
				chance = tf.reshape(chance, (self.batch_size, -1))
				target_chance = target_chance_batch[:, i, :]
				_, _, chance_loss = self.loss_function(None, None, chance, None, None, target_chance, False)
				total_chance_loss += scale_gradient(chance_loss, 1.0/self.num_unroll_steps)
			timers[5] += time.time()-st
			st=time.time()
			# Scale the value loss, paper recommends by 0.25 (See paper appendix Reanalyze)
			if self.config.PER:
				total_value_loss *= weight_batch
				total_reward_loss *= weight_batch
				total_chance_loss *= weight_batch
				total_policy_loss *= weight_batch
			loss = total_value_loss * self.config.loss_weights[0] + total_reward_loss * self.config.loss_weights[1] + total_chance_loss * self.config.loss_weights[2] + total_policy_loss * self.config.loss_weights[3]
			'''if self.config.PER:
				# Correct PER bias by using importance-sampling (IS) weights
				# this is why value_loss*weight+reward_loss+policy_loss != total_loss
				loss *= weight_batch'''

			timers[6] += time.time()-st
			st=time.time()
			# (Deepmind's pseudocode do sum, and werner-duvaud/muzero-general do mean. Both are the same.)
			loss = tf.math.reduce_mean(loss)#now loss is a scalar
			l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in self.model.trainable_variables])
			if step.value == 0:
				last_loss.set(loss.numpy())
				last_value_loss.set(tf.math.reduce_mean(total_value_loss).numpy())
				last_reward_loss.set(tf.math.reduce_mean(total_reward_loss).numpy())
				last_chance_loss.set(tf.math.reduce_mean(total_chance_loss).numpy())
				last_policy_loss.set(tf.math.reduce_mean(total_policy_loss).numpy())
				last_l2_loss.set(l2_loss.numpy())
				step.set(step.value+1)
			timers[7] += time.time()-st
			loss += l2_loss*self.l2_weight
			return loss

		# Optimize
		for _ in range(self.config.training_steps_per_batch):
			self.optimizer.minimize(loss_fn, self.model.trainable_variables)
			self.training_step += 1
			if self.training_step % self.config.checkpoint_interval == 0:
				shared_storage.save_weights(copy.deepcopy(self.model.get_weights()))
				shared_storage.save()
		print(f'''last_loss:{last_loss.value},
last_value_loss:{last_value_loss.value},
last_reward_loss:{last_reward_loss.value},
last_chance_loss:{last_chance_loss.value},
last_policy_loss:{last_policy_loss.value},
last_l2_loss:{last_l2_loss.value}''')
		return (
			priorities,
			# For log purpose
			last_loss.value,
			last_value_loss.value,
			last_reward_loss.value,
			last_chance_loss.value,
			last_policy_loss.value,
			last_l2_loss.value,

			last_value_initial.value,
			last_value_recurrent.value,
			last_reward.value,
			last_value_initial_delta.value,
			last_value_recurrent_delta.value,
			last_reward_delta.value,
		)

	def update_learning_rate(self):
		"""
		Update learning rate
		"""
		learning_rate = self.config.learning_rate_init * self.config.learning_rate_decay_rate ** (
			self.training_step / self.config.learning_rate_decay_steps
		)
		self.optimizer.learning_rate = learning_rate

	@staticmethod
	def loss_function(
		value,
		reward,
		policy_logits,
		target_value,
		target_reward,
		target_policy,
		using_support
	):
		# Cross-entropy seems to have a better convergence than MSE
		if using_support:
			value_loss = tf.nn.softmax_cross_entropy_with_logits(target_value, value) if value is not None else None
			reward_loss = tf.nn.softmax_cross_entropy_with_logits(target_reward, reward) if reward is not None else None
		else:
			value_loss = tf.keras.losses.MeanSquaredError()(target_value, value) if value is not None else None
			reward_loss = tf.keras.losses.MeanSquaredError()(target_reward, reward) if reward is not None else None
		policy_loss = tf.nn.softmax_cross_entropy_with_logits(target_policy, policy_logits) if policy_logits is not None else None
		#maybe I should add a loss function for chance
		return value_loss, reward_loss, policy_loss
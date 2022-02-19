import copy
import time

import numpy as np
import ray
import tensorflow as tf

import network
import my_config
def scale_gradient(tensor, scale):
	"""Scales the gradient for the backward pass."""
	return tensor * scale + tf.stop_gradient(tensor) * (1 - scale)

class Trainer:
	"""
	Class which run in a dedicated thread to train a neural network and save it
	in the shared storage.
	"""

	def __init__(self, initial_checkpoint, model:network.Network, config:my_config.Config):
		self.config = config
		seed=config.seed
		# Fix random generator seed
		np.random.seed(seed)
		tf.random.set_seed(seed)
		'''
		# Initialize the network
		self.model = network.Network(self.config)
		self.model.set_weights(copy.deepcopy(initial_checkpoint["weights"]))
		'''
		self.training_step = initial_checkpoint["training_step"]

		# Initialize the optimizer
		if self.config.optimizer == "SGD":
			self.optimizer = tf.keras.optimizers.SGD(
				learning_rate=self.config.learning_rate_init,
				momentum=self.config.momentum,
			)
		elif self.config.optimizer == "Adam":
			self.optimizer = tf.keras.optimizers.Adam(
				learning_rate=self.config.learning_rate_init,
			)
		else:
			raise NotImplementedError(
				f"{self.config.optimizer} is not implemented. You can change the optimizer manually in trainer.py."
			)

	def update_weights(self, replay_buffer, shared_storage):
		# Wait for the replay buffer to be filled
		assert shared_storage.get_info("num_played_games") > 0, 'no enough games, get 0'

		next_batch = replay_buffer.get_batch.remote()
		# Training loop
		shared_storage.clear_loss()
		while (self.training_step / max(
				1, ray.get(shared_storage.get_info("num_played_steps")))
				> self.config.training_steps_to_selfplay_steps_ratio
				and
				self.training_step < self.config.training_steps):
			index_batch, batch = ray.get(next_batch)
			next_batch = replay_buffer.get_batch.remote()
			self.update_learning_rate()
			(
				priorities,
				total_loss,
				value_loss,
				reward_loss,
				policy_loss,
			) = self.update_weights(batch, shared_storage)

			if self.config.PER:
				# Save new priorities in the replay buffer (See https://arxiv.org/abs/1803.00933)
				replay_buffer.update_priorities.remote(priorities, index_batch)

			# Save to the shared storage
			shared_storage.set_info(
				{
					"training_step": self.training_step,
					#"learning_rate": self.optimizer.learning_rate,
				}
			)
			shared_storage.append_loss(total_loss,value_loss,reward_loss,policy_loss)
			# Managing the self-play / training ratio
		# loss log:
		# clear losses saved in shared storage when training start. 
		# train until training to selfplay steps ratio is up to the config.

		# log has a long series of losses.
	def update_weights(self, batch, shared_storage):
		"""
		Perform one training step.
		"""
		(
			observation_batch,
			action_batch,
			target_value_batch,
			target_reward_batch,
			target_policy_batch,
			weight_batch,
		) = batch
		batchsize=action_batch.shape[0]
		# Keep values as scalars for calculating the priorities for the prioritized replay
		target_value_scalar = np.array(target_value_batch, dtype="float32")
		priorities = np.zeros_like(target_value_scalar)

		# observation_batch: batch, channels, height, width
		# action_batch: batch, num_unroll_steps+1
		# target_value: batch, num_unroll_steps+1
		# target_reward: batch, num_unroll_steps+1
		# target_policy: batch, num_unroll_steps+1, len(action_space)
		target_value_batch = network.scalar_to_support(target_value_batch, self.config.support)
		target_reward_batch = network.scalar_to_support(target_reward_batch, self.config.support)
		# target_value: batch, num_unroll_steps+1, 2*support_size+1
		# target_reward: batch, num_unroll_steps+1, 2*support_size+1

		## Generate predictions
		def loss_fn():
			value, reward, policy_logits, hidden_state = self.model.initial_inference(
				observation_batch
			)
			predictions = [(value, reward, policy_logits)]#type 0
			for i in range(1, action_batch.shape[1],2):
				_, _, _, hidden_state = self.model.recurrent_inference(
					hidden_state, action_batch[:, i]
				)
				hidden_state=scale_gradient(hidden_state, 0.5)
				value, reward, policy_logits, hidden_state = self.model.recurrent_inference(
					hidden_state, action_batch[:, i+1]
				)
				# Scale the gradient at the start of the dynamics function (See paper appendix Training)
				hidden_state=scale_gradient(hidden_state, 0.5)
				predictions.append((value, reward, policy_logits))
			# predictions: num_unroll_steps+1, 3, batch, 2*support_size+1 | 2*support_size+1 | 9 (according to the 2nd dim)
			total_value_loss,total_reward_loss,total_policy_loss=tf.zeros((batchsize))
			# shape of losses: batchsize
			for i, prediction in enumerate(predictions):
				value, reward, policy_logits = prediction
				target_value=target_value_batch[:,i,:]
				target_reward=target_reward_batch[:,i,:]
				target_policy=target_policy_batch[:,i,:]
				value_loss, reward_loss, policy_loss = self.loss_function(value,reward,policy_logits,target_value,target_reward,target_policy)
				if i==0:
					total_value_loss+=value_loss
					total_policy_loss+=policy_loss
				else:
					total_value_loss+=scale_gradient(value_loss, 1.0/(len(predictions)-1))
					total_reward_loss+=scale_gradient(reward_loss, 1.0/(len(predictions)-1))
					total_policy_loss+=scale_gradient(policy_loss, 1.0/(len(predictions)-1))
				pred_value_scalar = network.support_to_scalar(value, self.config.support_size)
				priorities[:, i] = (
					np.abs(pred_value_scalar - target_value_scalar[:, i])
					** self.config.PER_alpha
				)
			# Scale the value loss, paper recommends by 0.25 (See paper appendix Reanalyze)
			loss = total_value_loss * self.config.value_loss_weight + total_reward_loss + total_policy_loss
			if self.config.PER:
				# Correct PER bias by using importance-sampling (IS) weights
				loss *= weight_batch
			# (Deepmind's pseudocode do a sum, and werner-duvaud/muzero-general do a mean. Both are the same.) 
			loss=tf.math.reduce_mean(loss)#now loss is a scalar
			last_loss=loss.numpy()
			last_value_loss=tf.math.reduce_mean(total_value_loss).numpy()
			last_reward_loss=tf.math.reduce_mean(total_reward_loss).numpy()
			last_policy_loss=tf.math.reduce_mean(total_policy_loss).numpy()
			return loss

		# Optimize
		last_loss,last_value_loss,last_reward_loss,last_policy_loss=[None]*4
		for _ in range(self.config.steps_per_batch):
			self.optimizer.minimize(loss_fn,self.model.trainable_variables)
			self.training_step += 1

			if self.training_step % self.config.checkpoint_interval == 0:
				shared_storage.save_weights(copy.deepcopy(self.model.get_weights()))
				shared_storage.save()
		return (
			priorities,
			# For log purpose
			last_loss,
			last_value_loss,
			last_reward_loss,
			last_policy_loss,
		)

	def update_learning_rate(self):
		"""
		Update learning rate
		"""
		learning_rate = self.config.learning_rate_init * self.config.learning_rate_decay_rate ** (
			self.training_step / self.config.learning_rate_decay_steps
		)
		self.optimizer.learning_rate=learning_rate

	@staticmethod
	def loss_function(
		value,
		reward,
		policy_logits,
		target_value,
		target_reward,
		target_policy,
	):
		# Cross-entropy seems to have a better convergence than MSE
		value_loss = tf.nn.softmax_cross_entropy_with_logits(target_value,value)
		reward_loss = tf.nn.softmax_cross_entropy_with_logits(target_reward,reward)
		policy_loss = tf.nn.softmax_cross_entropy_with_logits(target_policy,policy_logits)
		return value_loss, reward_loss, policy_loss

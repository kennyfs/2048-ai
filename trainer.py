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
		self.model=model
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

	def run_update_weights(self, replay_buffer, shared_storage):
		# Wait for the replay buffer to be filled
		assert shared_storage.get_info("num_played_games") > 0, 'no enough games, get 0'

		next_batch = replay_buffer.get_batch.remote()
		# Training loop
		shared_storage.clear_loss()
		while (self.training_step / max(
				1, shared_storage.get_info("num_played_steps"))
				< self.config.training_steps_to_selfplay_steps_ratio
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
		) = batch#batches are all np.array
		#for debug, check the shape of the batch, maybe is useful
		for b in batch:
			print(b.shape)
		batchsize=action_batch.shape[0]
		num_unroll_steps=action_batch.shape[1]
		# Keep values as scalars for calculating the priorities for the prioritized replay
		target_value_scalar = np.copy(target_value_batch)
		priorities = np.zeros_like(target_value_scalar)

		# observation_batch: batch, channels, height, width
		# action_batch: batch, num_unroll_steps
		# target_value: batch, num_unroll_steps/2+1 #+1 for the those of initial reference
		# target_reward: batch, num_unroll_steps/2+1
		# target_policy: batch, num_unroll_steps/2+1, action_space_size
		target_value_batch = network.scalar_to_support(target_value_batch, self.config.support)
		target_reward_batch = network.scalar_to_support(target_reward_batch, self.config.support)
		# target_value: batch, num_unroll_steps/2+1, 2*support+1
		# target_reward: batch, num_unroll_steps/2+1, 2*support+1
		class tmp:#I don't know better solution
			def __init__(self, value=None):
				self.value=value
			def set(self, value):
				self.value=value
		last_loss=tmp()
		last_value_loss=tmp()
		last_reward_loss=tmp()
		last_policy_loss=tmp()
		## Generate predictions
		def loss_fn():
			output=self.model.initial_inference(
				observation_batch
			)#defined as time 0
			hidden_state=output.hidden_state
			value=output.value
			reward=np.zeros_like(value)
			policy_logits=output.policy
			predictions = [(value, reward, policy_logits)]
			for i in range(0, action_batch.shape[1],2):### start to check data processing here
				hidden_state = self.model.recurrent_inference(
					hidden_state, action_batch[:, i]
				).hidden_state
				hidden_state=scale_gradient(hidden_state, 0.5)
				output = self.model.recurrent_inference(
					hidden_state, action_batch[:, i+1]
				)
				reward=output.reward
				hidden_state=output.hidden_state
				value=output.value
				policy_logits=output.policy
				# Scale the gradient at the start of the dynamics function (See paper appendix Training)
				hidden_state=scale_gradient(hidden_state, 0.5)
				predictions.append((value, reward, policy_logits))
			assert len(predictions)==num_unroll_steps/2+1,f'len(predictions):{len(predictions)}\nnum_unroll_steps:{num_unroll_steps}'
			#maybe there should be more assertion
			
			#predictions[t]=output at time t*2
			# predictions: num_unroll_steps/2+1, 3, (batch, (2*support+1 | 2*support+1 | 9)) (according to the 2nd dim)
			total_value_loss,total_reward_loss,total_policy_loss=tf.zeros((batchsize)),tf.zeros((batchsize)),tf.zeros((batchsize))
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
					total_value_loss+=scale_gradient(value_loss, 1.0/(num_unroll_steps/2-1))
					total_reward_loss+=scale_gradient(reward_loss, 1.0/(num_unroll_steps/2-1))
					total_policy_loss+=scale_gradient(policy_loss, 1.0/(num_unroll_steps/2-1))
				pred_value_scalar = network.support_to_scalar(value, self.config.support)
				priorities[:, i] = (
					np.abs(pred_value_scalar - target_value_scalar[:, i])
					** self.config.PER_alpha
				)
			# Scale the value loss, paper recommends by 0.25 (See paper appendix Reanalyze)
			loss = total_value_loss * self.config.loss_weights[0] + total_reward_loss * self.config.loss_weights[1] + total_policy_loss * self.config.loss_weights[2]
			if self.config.PER:
				# Correct PER bias by using importance-sampling (IS) weights
				# this is why value_loss*weight+reward_loss+policy_loss != total_loss
				loss *= weight_batch
			# (Deepmind's pseudocode do sum, and werner-duvaud/muzero-general do mean. Both are the same.) 
			loss=tf.math.reduce_mean(loss)#now loss is a scalar
			last_loss.set(loss.numpy())
			last_value_loss.set(tf.math.reduce_mean(total_value_loss).numpy())
			last_reward_loss.set(tf.math.reduce_mean(total_reward_loss).numpy())
			last_policy_loss.set(tf.math.reduce_mean(total_policy_loss).numpy())
			return loss

		# Optimize
		for _ in range(self.config.training_steps_per_batch):
			self.optimizer.minimize(loss_fn,self.model.trainable_variables)
			self.training_step += 1
			if self.training_step % self.config.checkpoint_interval == 0:
				shared_storage.save_weights(copy.deepcopy(self.model.get_weights()))
				shared_storage.save()
		return (
			priorities,
			# For log purpose
			last_loss.value,
			last_value_loss.value,
			last_reward_loss.value,
			last_policy_loss.value,
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

import asyncio
import collections
import random
from abc import ABC, abstractmethod
import time

import numpy as np
import tensorflow as tf

import my_config


def scale_hidden_state(to_scale_hidden_state:np.array):
	'''
	input should contain batch size, but it can whether be a matrix or not.
	shape:(batchsize, hidden_state_size) or (batchsize, hidden_state_size_x, hidden_state_size_y)
	'''
	shape = to_scale_hidden_state.shape
	hidden_state_as_matrix = (len(shape) == 4)
	if hidden_state_as_matrix:
		to_scale_hidden_state = tf.reshape(to_scale_hidden_state, (shape[0], -1))#flatten from [batch_size, hidden_state_size_x, hidden_state_size_y] to [batch_size, hidden_state_size_x*hidden_state_size_y]
	min_encoded_state = tf.math.reduce_min(to_scale_hidden_state, axis = 1, keepdims = True)
	max_encoded_state = tf.math.reduce_max(to_scale_hidden_state, axis = 1, keepdims = True)
	scale_encoded_state = max_encoded_state - min_encoded_state
	scale_encoded_state = tf.where(scale_encoded_state < 1e-5, scale_encoded_state+1e-5, scale_encoded_state)#avoid divided by 0 or too small value
	encoded_state_normalized = (to_scale_hidden_state - min_encoded_state) / scale_encoded_state
	if hidden_state_as_matrix:
		encoded_state_normalized = tf.reshape(encoded_state_normalized, shape)#and reshape to original shape
	return encoded_state_normalized
	
@tf.function
def support_to_scalar(logits, support_size, from_logits = True):# logits is in shape (batch_size, full_support_size)
	"""
	Transform a categorical representation to a scalar
	See paper appendix F Network Architecture (P.14)
	"""
	# Decode to a scalar
	if support_size == 0:
		return logits
	if from_logits:
		probabilities = tf.nn.softmax(logits, axis = -1)
	else:
		probabilities = logits
	support = tf.range(-support_size, support_size+1, delta = 1, dtype = tf.float32)
	support = tf.expand_dims(support, axis = 0)# in shape (1, full_support_size)
	support = tf.tile(support, (probabilities.shape[0], 1))
	x = tf.reduce_sum(support * probabilities, axis = 1)
	# Invert the scaling (defined in https://arxiv.org/abs/1805.11593)
	x = tf.math.sign(x) * (
		((tf.math.sqrt(1 + 4 * 0.001 * (tf.math.abs(x) + 1 + 0.001)) - 1) / (2 * 0.001))
		** 2
		- 1
	)
	return x
def scalar_to_support(x, support_size):
	"""
	Transform a scalar to a categorical representation with (2 * support_size + 1) categories
	See paper appendix Network Architecture
	shape of input is (batch, num_unroll_steps+1)
	"""
	# Reduce the scale (defined in https://arxiv.org/abs/1805.11593)
	original_shape = tf.shape(x)
	x = tf.reshape(x, (-1))#flatten
	length = x.shape[0]
	x = tf.math.sign(x) * (tf.math.sqrt(tf.math.abs(x) + 1) - 1) + 0.001 * x
	# Encode on a vector
	x = tf.clip_by_value(x, -support_size, support_size)
	floor = tf.math.floor(x)
	prob = x - floor
	#target: set floor to be 1-prob
	#		   and floor+1 to be prob
	floor = tf.cast(floor, 'int32')
	logits = tf.zeros((length * (2 * support_size + 1)))#flattened of (length , 2 * support_size + 1)
	ori_indices = floor+support_size
	indices_to_add = tf.range(length)*(2 * support_size + 1)
	indices = ori_indices+indices_to_add
	indices = tf.expand_dims(indices, axis = -1)# index is in 1-dimensional
	logits = tf.tensor_scatter_nd_update(
		logits, indices = indices, updates = 1 - prob
	)
	ori_indices = ori_indices+1
	prob = tf.where(2 * support_size < ori_indices, 0., prob)
	ori_indices = tf.where(2 * support_size < ori_indices, 0, ori_indices)
	indices = ori_indices+indices_to_add
	indices = tf.expand_dims(indices, axis = -1)# index is in 1-dimensional
	logits = tf.tensor_scatter_nd_update(
		logits, indices = indices, updates = prob
	)
	logits = tf.reshape(logits, (*original_shape, -1))
	return logits
def action_to_onehot(action, board_size):
	only_one = False
	if len(action.shape) == 0:
		only_one = True
		action = tf.cast(tf.expand_dims(action, axis = 0), dtype = tf.int32)#treat it as (1)
	batch_size = action.shape[0]
	indices = tf.concat((tf.expand_dims(tf.range(batch_size), axis = -1), tf.expand_dims(action, axis = -1)), axis = -1)
	# action: (batch)
	# output: (batch, 4, board_size, board_size)
	# indices: (batch, 2)# batch (tf.range(batch)), action concatenated
	# updates: tf.ones((batch, board_size, board_size))/ (board_size**2)
	ret = tf.scatter_nd(indices, tf.ones((batch_size, board_size, board_size))/(board_size**2), (batch_size, 4, board_size, board_size))
	if only_one:
		ret = tf.reshape(ret, (4, board_size, board_size))
	return ret
@tf.function
def random_action_to_onehot(action, board_size):#defaulted for matrix
	only_one = False
	if len(action.shape) == 0:
		only_one = True
		action = tf.cast(tf.expand_dims(action, axis = 0), dtype = tf.int32)#treat it as (1)
	batch_size = action.shape[0]
	action -= 4
	# action: (batch)
	# output: (batch, 2, board_size, board_size) #reshaped from (batch, 2*board_size**2)
	# indices: (batch, 2) #tf.range(batch) concatenated with action, resulting in something like (0,4), (1,5)
	indices = tf.concat((tf.expand_dims(tf.range(batch_size), 1), tf.expand_dims(action, 1)), axis = 1)
	ret = tf.scatter_nd(indices, tf.ones((batch_size)), (batch_size, 2*board_size**2))
	ret = tf.reshape(ret, (batch_size, 2, board_size, board_size))
	if only_one:
		ret = tf.reshape(ret, (2, board_size, board_size))
	return ret
@tf.function
def softmax_chance(chance):
	#batched
	assert len(chance.shape) == 4
	board_size = chance.shape[2]
	batch = chance.shape[0]
	if chance.shape[1:] == [1, board_size, board_size]:
		chance = tf.reshape(chance, (batch, board_size**2))
		chance = tf.nn.softmax(chance, axis = -1)
		chance = tf.concat([np.zeros([batch,4]),chance*0.9, chance*0.1], axis = 1)
	else:
		chance = tf.reshape(chance, (batch, 2*board_size**2))
		chance = tf.nn.softmax(chance, axis = -1)
		chance = tf.concat([np.zeros([batch,4]), chance], axis = 1)
	#output is (batch, 4+2*board_size**2)
	assert chance.shape == (batch, 4+2*board_size**2)
	return chance
NetworkOutput = collections.namedtuple('NetworkOutput', ['reward', 'hidden_state', 'value', 'policy'])

												####shapes:###
#observation:		in shape of (batch_size, channels, board_size_x, board_size_y)#board_size_x = board_size_y in most cases
#channels = history_length*planes per image
#hidden_state:		in shape of (batch_size, hidden_state_size(defined in my_config.py))
#					or (batch_size, num_channels, boardsize, boardsize)
#action:			if one-hotted, for fully connected network in shape of (batch_size, 4)
#					if one-hotted, for resnet in shape of (batch_size, 4, boardsize, boardsize)#4 for UDLR, all 1 in selected plane
#policy:			in shape of (batch_size, 4(UDLR))
#value and reward:	in shape of (batch_size, full_support_size) if using support, else (batch_size, 1) #about "support", described in config.py
def my_adjust_dims(inputs, expected_shape_length, axis = 0):
	'''
	extend from last and remove from first by default
	axis: 0:default, 1:first, 2:last
	'''
	axises = len(inputs.shape)
	delta = axises-expected_shape_length
	if delta < 0:
		if axis == 0 or axis == 1:
			inputs = np.expand_dims(inputs, axis = list(range(-delta)))
		else:
			inputs = np.expand_dims(inputs, axis = list(range(delta, 0)))
	if delta > 0:
		if axis == 0 or axis == 2:
			inputs = inputs.reshape((*inputs.shape[:-delta-1], -1))
		else:
			inputs = inputs.reshape((-1, *inputs.shape[delta+1:]))
	return inputs
class AbstractNetwork(ABC):
	'''
	If directly call initial and recurrent inference of Network's,
	it means directly get result, without queuing(so with batch size).
	(This is for training)
	'''
	def __init__(self):
		super().__init__()
		pass
		
	@abstractmethod
	def representation(self, observation):
		#output:hidden_state
		#hidden state is scaled
		pass
		
	@abstractmethod
	def dynamics(self, hidden_state, action, chance):
		#output: hidden_state, reward
		#hidden state is scaled
		#reward is in logits
		pass

	@abstractmethod
	def chance(self, hidden_state, action):
		'''
		output: chance(batch, 1, 4, 4)
		actually, it outputs where to put the next tile(blanks)
		'''
		pass

	@abstractmethod
	def prediction(self, hidden_state):
		#output: policy, value
		#value is in logits
		pass
	
	def initial_inference(self, observation) -> NetworkOutput:
		'''
		directly inference, for training and reanalyze
		input shape: batch, channel, width, height
		'''
		assert len(observation.shape) == 4
		hidden_state = self.representation(observation)
		#hidden_state:batch, hidden_size
		policy, value = self.prediction(hidden_state)
		#policy:batch, 4
		#value:batch, 1 if not support, else batch, support*2+1
		return NetworkOutput(policy = policy, value = value, reward = None, hidden_state = hidden_state)
	def recurrent_inference(self, hidden_state, action_batch, random_action_batch, onehotted = True, matrix = True) -> NetworkOutput:
		'''
		directly inference, for training
		'''
		#auto detect matrix
		if not matrix and len(hidden_state.shape) == 4:
			matrix = True
		if not onehotted:
			action_batch = action_to_onehot(action_batch, self.config.board_size)
			random_action_batch = random_action_to_onehot(random_action_batch, self.config.board_size)
		hidden_state, reward = self.dynamics([hidden_state, action_batch, random_action_batch])
		hidden_state = scale_hidden_state(hidden_state)
		policy, value = self.prediction(hidden_state)
		return NetworkOutput(policy = policy, value = value, reward = reward, hidden_state = hidden_state)
	@abstractmethod
	def get_weights(self):
		pass
	@abstractmethod
	def set_weights(self, weights):
		pass
	@abstractmethod
	def summary(self):
		pass
class Network:
	def __new__(cls, config):
		if config.network_type == "resnet":
			return ResNetNetwork(config)
		else:
			raise NotImplementedError

##################################
########## CNN or RESNET #########

def conv3x3(out_channels, stride = 1):
	return tf.keras.layers.Conv2D(out_channels, kernel_size = 3, strides = stride, padding = 'same', use_bias = True, data_format = 'channels_first')
def conv1x1(out_channels, stride = 1):
	return tf.keras.layers.Conv2D(out_channels, kernel_size = 1, strides = stride, padding = 'same', use_bias = True, data_format = 'channels_first')
class Squeeze_excitation_block(tf.keras.Model):
	def __init__(self, filter_sq, num_channels):
		super().__init__()
		self.filter_sq = filter_sq
		self.pool = tf.keras.layers.GlobalAveragePooling2D(data_format = 'channels_first')
		self.dense_1 = tf.keras.layers.Dense(filter_sq)
		self.dense_2 = tf.keras.layers.Dense(num_channels)
		self.relu = tf.keras.layers.Activation('relu')
		self.sigmoid = tf.keras.layers.Activation('sigmoid')
		self.reshape = tf.keras.layers.Reshape((num_channels, 1, 1))
	def call(self, x):
		squeezed = self.pool(x)

		excitation = self.dense_1(squeezed)
		excitation = self.relu(excitation)
		excitation = self.dense_2(excitation)
		excitation = self.sigmoid(excitation)
		excitation = self.reshape(excitation)

		scale = x * excitation
		return scale
class ResidualBlock(tf.keras.Model):
	def __init__(self, num_channels):
		super().__init__()
		self.conv1 = conv3x3(num_channels, num_channels)
		self.conv2 = conv3x3(num_channels, num_channels)
		self.bn1 = tf.keras.layers.BatchNormalization()
		self.bn2 = tf.keras.layers.BatchNormalization()
		self.relu = tf.keras.layers.ReLU()
		self.se = Squeeze_excitation_block(num_channels, num_channels)
	def call(self, x):
		residual = x
		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)
		out = self.conv2(out)
		out = self.bn2(out)
		out = self.se(out)
		out += residual
		out = self.relu(out)
		return out

#no need to downsample because 2048 is only 4x4
'''
representation_model:
	input:observation
	output:hidden_state
dynamic_model:
	input:hidden_state, action, random_action_distribution
	output:new_hidden_state, reward, random_action_distribution
representation_model:
	input:hidden_state
	output:policy, value
'''
class representation(tf.keras.Model):
	def __init__(self, input_shape, num_channels, num_blocks):
		super().__init__()
		self.conv = conv3x3(num_channels)
		self.bn = tf.keras.layers.BatchNormalization()
		self.relu = tf.keras.layers.ReLU()
		self.resblock = [ResidualBlock(num_channels) for i in range(num_blocks)]
		#self.resblock = [[conv3x3(num_channels, num_channels), tf.keras.layers.BatchNormalization(), conv3x3(num_channels, num_channels), tf.keras.layers.BatchNormalization()]for i in range(num_blocks)]
		self.build(input_shape)

	def call(self, x):
		out = self.conv(x)
		out = self.bn(out)
		out = self.relu(out)
		for block in self.resblock:
			out = block(out)
		return out

class dynamics(tf.keras.Model):
	def __init__(self,
		input_shape,
		num_channels,
		num_blocks_1,
		num_blocks_2,
		reduced_channels_reward,
		reward_layers,
		support):
		super().__init__()
		self.conv = conv3x3(num_channels)
		self.bn = tf.keras.layers.BatchNormalization()
		self.relu = tf.keras.layers.ReLU()
		self.resblock_1 = [ResidualBlock(num_channels) for i in range(num_blocks_1)]
		#self.resblock = [[conv3x3(num_channels, num_channels), tf.keras.layers.BatchNormalization(), conv3x3(num_channels, num_channels), tf.keras.layers.BatchNormalization()]for i in range(num_blocks)]
		
		self.conv_reward = conv1x1(reduced_channels_reward)
		self.bn_reward = tf.keras.layers.BatchNormalization()
		self.flatten = tf.keras.layers.Flatten()

		self.reward_output = [tf.keras.layers.Dense(size, activation = 'relu') for size in reward_layers]+[tf.keras.layers.Dense(support*2+1)]

		self.conv_2 = conv3x3(num_channels)
		self.bn_2 = tf.keras.layers.BatchNormalization()

		self.resblock_2 = [ResidualBlock(num_channels) for i in range(num_blocks_2)]
		self.build(input_shape)

	def call(self, x):
		hidden_state, action, random_action = x
		hidden_state_and_action = tf.keras.layers.concatenate([hidden_state, action], axis = 1)
		out = self.conv(hidden_state_and_action)
		out = self.bn(out)
		out = self.relu(out)
		for block in self.resblock_1:
			out = block(out)
		hidden_state = out
		reward = hidden_state
		reward = self.conv_reward(reward)
		reward = self.bn_reward(reward)
		reward = self.flatten(reward)
		reward = self.relu(reward)
		for layer in self.reward_output:
			reward = layer(reward)

		#try to resnet to output reward, as reward isn't related to the random action
		#it may not work well(there are fewer layers before reward)
		hidden_state = tf.keras.layers.concatenate([hidden_state, random_action], axis = 1)
		hidden_state = self.conv_2(hidden_state)
		hidden_state = self.bn_2(hidden_state)
		hidden_state = self.relu(hidden_state)
		for block in self.resblock_2:
			hidden_state = block(hidden_state)
		return hidden_state, reward

class chance(tf.keras.Model):
	def __init__(self,
		input_shape,
		num_channels,
		num_blocks,
		output_filter):
		super().__init__()
		self.conv = conv3x3(num_channels)
		self.bn = tf.keras.layers.BatchNormalization()
		self.relu = tf.keras.layers.ReLU()
		self.resblock = [ResidualBlock(num_channels) for _ in range(num_blocks)]
		self.conv_out = conv3x3(output_filter)

		self.build(input_shape)
	def call(self, x):
		#x contains hidden_state and action
		hidden_state, action = x
		x = tf.keras.layers.concatenate([hidden_state, action], axis = 1)
		out = self.conv(x)
		out = self.bn(out)
		out = self.relu(out)
		for block in self.resblock:
			out = block(out)
		out = self.conv_out(out)
		return out
class prediction(tf.keras.Model):
	def __init__(self,
		input_shape,
		action_space_size,
		reduced_channels_value,
        reduced_channels_policy,
		value_layers,
		policy_layers,
		support):
		super().__init__()
		self.conv1x1_value = conv1x1(reduced_channels_value)
		self.conv1x1_policy = conv1x1(reduced_channels_policy)
		self.flatten = tf.keras.layers.Flatten()
		self.dense_value = [tf.keras.layers.Dense(size, activation = 'relu') for size in value_layers]+[tf.keras.layers.Dense(support*2+1)]
		self.dense_policy = [tf.keras.layers.Dense(size, activation = 'relu') for size in policy_layers]+[tf.keras.layers.Dense(action_space_size)]
		self.build(input_shape)

	def call(self, x):
		out = self.conv1x1_policy(x)
		out = self.flatten(out)
		for layer in self.dense_policy:
			out = layer(out)
		policy = out

		out = self.conv1x1_value(x)
		out = self.flatten(out)
		for layer in self.dense_value:
			out = layer(out)
		value = out
		return policy, value

class ResNetNetwork(AbstractNetwork):
	def __init__(self, config:my_config.Config):
		super().__init__()
		self.config = config
		self.representation_model = representation(
			[None]+config.observation_shape,
			config.num_channels,
			config.num_blocks-1)
		self.dynamics_model = dynamics(
			[[None, config.num_channels, config.board_size, config.board_size], [None, 4, config.board_size, config.board_size], [None, 2, config.board_size, config.board_size]], #hidden_state(num_channels) + action(4) + random_action(2)
			config.num_channels,
			config.num_blocks_dynamic_1-1,
			config.num_blocks_dynamic_2-1,
			config.reduced_channels_reward,
			config.reward_layers,
			config.support, )
		self.prediction_model = prediction(
			[None, config.num_channels, config.board_size, config.board_size], #hidden_state(num_channels)
			4,
			config.reduced_channels_value,
			config.reduced_channels_policy,
			config.value_layers,
			config.policy_layers,
			config.support)
		self.chance_model = chance(
			[[None, config.num_channels, config.board_size, config.board_size], [None, 4, config.board_size, config.board_size]], #hidden_state(num_channels) + action(4)
			config.num_channels,
			config.num_blocks-1,
			config.chance_output
		)
		self.trainable_variables = self.representation_model.trainable_variables+self.dynamics_model.trainable_variables+self.chance_model.trainable_variables+self.prediction_model.trainable_variables
	def representation(self, observation):
		return scale_hidden_state(self.representation_model(observation))
	def dynamics(self, input):
		'''
		batched input, input is always [hidden_state, action, random_action]
		'''
		hidden_state, reward = self.dynamics_model(input)
		hidden_state = scale_hidden_state(hidden_state)
		return hidden_state, reward
	def chance(self, input):
		'''
		batched input, input is always [hidden_state, action]
		'''
		return self.chance_model(input)
	def prediction(self, hidden_state):
		policy, value = self.prediction_model(hidden_state)
		return policy, value
	def get_weights(self):
		return {'representation':self.representation_model.get_weights(), 'dynamics':self.dynamics_model.get_weights(), 'prediction':self.prediction_model.get_weights(), 'chance':self.chance_model.get_weights()}
	def set_weights(self, weights):
		'''
		set weights of 3 networks
		weights can be {name(str):weights(np.ndarray)}
		or list of 3 weights [weights(np.ndarray)]
		'''
		if isinstance(weights, dict):
			for name, weight in weights.items():
				if name == 'representation':
					self.representation_model.set_weights(weight)
				elif name == 'dynamics':
					self.dynamics_model.set_weights(weight)
				elif name == 'prediction':
					self.prediction_model.set_weights(weight)
				elif name == 'chance':
					self.chance_model.set_weights(weight)
				else:
					raise NotImplementedError
		elif isinstance(weights, list):
			self.representation_model.set_weights(weights[0])
			self.dynamics_model.set_weights(weights[1])
			self.prediction_model.set_weights(weights[2])
			self.chance_model.set_weights(weights[3])
		else:
			raise NotImplementedError
	def summary(self):
		ret = ''
		for model in self.representation_model, self.dynamics_model, self.prediction_model, self.chance_model:
			stringlist = []
			model.summary(print_fn = lambda x: stringlist.append(x))
			ret += "\n".join(stringlist)+'\n-----------------------------\n'
		return ret
####### End CNN or RESNET ########
##################################

QueueItem = collections.namedtuple("QueueItem", ['inputs', 'future'])
class Manager:
	'''
	Queuing requests of network prediction, and run them together to improve efficiency
	I really feel the difference
	
	input to each network for a single prediction should be in [*expected_shape], rather than [batch_size(1), *expected_shape]
		process in self.prediction_worker
	
	policy is not softmaxed
	value and reward are scalars
	chance is softmaxed
	'''
	def __init__(self, config:my_config.Config, model:AbstractNetwork):
		self.config=config
		self.support = config.support
		self.queue = config.manager_queue
		self.loop = asyncio.get_event_loop()
		#callable model
		self.model = model
		self.representation = model.representation
		self.dynamics = model.dynamics
		self.prediction = model.prediction
		self.chance = model.chance
		if self.queue:
			self.representation_queue = asyncio.queues.Queue(config.model_max_threads)
			self.dynamics_queue = asyncio.queues.Queue(config.model_max_threads)
			self.prediction_queue = asyncio.queues.Queue(config.model_max_threads)
			self.chance_queue = asyncio.queues.Queue(config.model_max_threads)
			
			self.coroutine_list = [self.prediction_worker()]
		else:
			self.coroutine_list = []
	async def push_queue(self, input:np.array, network:int):#network means which to use. If passing string consumes too much time, pass int instead.
		if self.queue:
			future = self.loop.create_future()
			item = QueueItem(input, future)
			if network == 0:#representation
				await self.representation_queue.put(item)
			elif network == 1:#dynamics
				await self.dynamics_queue.put(item)
			elif network == 2:#prediction
				await self.prediction_queue.put(item)
			elif network == 3:#chance
				await self.chance_queue.put(item)
			else:
				raise NotImplementedError
			return future
		else:
			if isinstance(input, list):
				input = [np.expand_dims(i, 0) for i in input]
			else:
				input = np.expand_dims(input, axis = 0)
			#sorry for ugly code
			with tf.device('/device:GPU:0'):
				if network == 0:#representation
					ret = self.representation(input)
					return ret[0]
				elif network == 1:#dynamics
					ret = self.dynamics(input)
					return ret[0][0], support_to_scalar(ret[1], self.support)[0]
				elif network == 2:#prediction
					ret = self.prediction(input)
					return ret[0][0], support_to_scalar(ret[1], self.support)[0]
				elif network == 3:#chance
					ret = self.chance(input)
					return softmax_chance(ret)[0]
				else:
					raise NotImplementedError
	def add_coroutine_list(self, toadd):
		#if toadd not in self.coroutine_list:
		self.coroutine_list.append(toadd)
			
	def run_coroutine_list(self, output = False):
		ret = self.loop.run_until_complete(asyncio.gather(*(self.coroutine_list)))
		if self.queue:
			self.coroutine_list = [self.prediction_worker()]
			if output:
				return ret[1:]
		else:
			self.coroutine_list = []
			if output:
				return ret
	def are_all_queues_empty(self):
		for q in (self.representation_queue, self.dynamics_queue, self.prediction_queue, self.chance_queue):
			if not q.empty():
				return False
		return True
	def get_weights(self):
		return self.model.get_weights()
	def set_weights(self, weights):
		self.model.set_weights(weights)
	async def prediction_worker(self):
		"""For better performance, queue prediction requests and predict together in this worker.
		speed up about 3x.
		"""
		margin = 10  # avoid finishing before other searches starting.
		while margin > 0:
			if self.are_all_queues_empty():
				await asyncio.sleep(1e-3)
				if self.are_all_queues_empty():
					margin -= 1
					await asyncio.sleep(1e-3)
				continue
			for id, queue, func in zip(range(4), (self.representation_queue, self.dynamics_queue, self.prediction_queue, self.chance_queue), (self.representation, self.dynamics, self.prediction, self.chance)):
				if queue.empty():
					continue
				item_list = [queue.get_nowait() for _ in range(queue.qsize())]  # type: list[QueueItem]
				if isinstance(item_list[0].inputs, list):
					input = [np.concatenate([np.expand_dims(item.inputs[i], axis = 0) for item in item_list], axis = 0) for i in range(len(item_list[0].inputs))]
				else:
					input = np.concatenate([np.expand_dims(item.inputs, axis = 0) for item in item_list], axis = 0)
				batch_size = len(item_list)
				#print('nn', batch_size)
				with tf.device('/device:GPU:0'):
					#start = time()
					results = func(input)
					#print('inference:', time()-start)
				if id == 0:
					hidden_state = results
					assert hidden_state.shape[0] == batch_size, 'sizes of hidden_state('+hidden_state.shape+') and item_list('+batch_size+') don\'t match, this should never happen.'
					for i in range(batch_size):
						item_list[i].future.set_result(hidden_state[i])
				elif id == 1:
					hidden_state, reward = results
					assert hidden_state.shape[0] == batch_size and reward.shape[0] == batch_size, 'sizes of hidden_state('+hidden_state.shape+'), reward('+reward.shape+'), and item_list('+batch_size+') don\'t match, this should never happen.'
					if self.support:
						reward = support_to_scalar(reward, self.support, True)
					else:
						reward = tf.reshape(reward, (-1))
					for i in range(batch_size):
						item_list[i].future.set_result((hidden_state[i], reward[i]))
						if random.random() < 0.02:
							print(f'r:{reward[i]}')
				elif id == 2:
					policy, value = results
					assert policy.shape[0] == batch_size and value.shape[0] == batch_size, 'sizes of policy('+policy.shape+'), value('+value.shape+'), and item_list('+batch_size+') don\'t match, this should never happen.'
					if self.support:
						value = support_to_scalar(value, self.support, True)
					else:
						value = tf.reshape(value, (-1))
					for i in range(batch_size):
						item_list[i].future.set_result((policy[i], value[i]))
						if random.random() < 0.02:
							print(f'v:{value[i]}')
				elif id == 3:
					chance = results
					assert chance.shape[0] == batch_size, 'sizes of chance('+chance.shape+'), and item_list('+batch_size+') don\'t match, this should never happen.'
					chance = softmax_chance(chance)
					assert chance.shape == (batch_size, 4+2*self.config.board_size**2), f'chance.shape:{chance.shape}'
					for i in range(batch_size):
						item_list[i].future.set_result(chance[i])
class Predictor:
	'''
	queuing and predict with manager
	'''
	def __init__(self, manager:Manager, config:my_config.Config):
		self.network_type = config.network_type#'resnet'/'fullyconnected'/...
		self.queue = config.manager_queue
		self.manager = manager
		self.push_queue = manager.push_queue
		self.config = config
		
	async def get_outputs(self, inputs, network:int):
		if self.queue:
			future = await self.push_queue(inputs, network)
			await future
			return future.result()
		else:
			return await self.push_queue(inputs, network)
	async def initial_inference(self, observation) -> NetworkOutput:
		'''
		input shape:
			observation: whether flattened or not and without batchsize
			it will be flattened in tf.keras.layers.Flatten
		'''
		hidden_state = await self.get_outputs(observation, 0)#already scaled
		policy, value = await self.get_outputs(hidden_state, 2)
		return NetworkOutput(reward = 0, hidden_state = hidden_state, value = value, policy = policy)

	async def recurrent_inference(self, hidden_state:np.array, action:int, random_action:int, onehotted:bool=False, debug:bool=False) -> NetworkOutput:
		'''
		input shape:
			all be without batchsize
			hidden_state:(hidden_state_size(32))
			action:(1) or ()
			random_action:(1) or ()

			for resnet
			hidden_state:(num_channels, hidden_state_size_x, hidden_state_size_y)
		'''
		assert isinstance(action, (int, np.int64)), f'action:{action}, type:{type(action)}'
		if not onehotted:
			action = action_to_onehot(action, self.config.board_size)
			random_action = random_action_to_onehot(random_action, self.config.board_size)
		new_hidden_state, reward = await self.get_outputs([hidden_state, action, random_action], 1)#hidden state is already scaled
		if debug:
			print(f'reward:{reward}')
		policy, value = await self.get_outputs(new_hidden_state, 2)
		
		return NetworkOutput(reward = reward, hidden_state = new_hidden_state, value = value, policy = policy)
	'''def run_coroutine_list(self):
		return self.manager.run_coroutine_list()
	def add_coroutine_list(self, toadd):
		self.manager.add_coroutine_list(toadd)#if efficiency is too low, directly append to the list'''
	async def chance(self, hidden_state:np.array, action:int, onehotted = False, debug:bool=False) -> np.array:
		if not onehotted:
			assert isinstance(action, (int, np.int64, np.int32)), f'action:{action}, type:{type(action)}'
			action = action_to_onehot(action, self.config.board_size)
		chance = await self.get_outputs([hidden_state, action], 3)#hidden state is already scaled
		if debug:
			print(chance)
		return chance
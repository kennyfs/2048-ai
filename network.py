import asyncio
import collections
from abc import ABC, abstractmethod
from copy import deepcopy as dc
from time import time

import numpy as np
import tensorflow as tf


def scale_hidden_state(to_scale_hidden_state:np.array):
	'''
	input should contain batch size, but it can whether be a matrix or not.
	shape:(batchsize, hidden_state_size) or (batchsize, hidden_state_size_x, hidden_state_size_y)
	'''
	shape=to_scale_hidden_state.shape
	hidden_state_as_matrix=len(shape)==3
	if hidden_state_as_matrix:
		to_scale_hidden_state=tf.reshape(to_scale_hidden_state,(shape[0],-1))#flatten from [batch_size,hidden_state_size_x,hidden_state_size_y] to [batch_size,hidden_state_size_x*hidden_state_size_y]
	min_encoded_state = tf.math.reduce_min(to_scale_hidden_state,axis=1,keepdims=True)
	max_encoded_state = tf.math.reduce_max(to_scale_hidden_state,axis=1,keepdims=True)
	scale_encoded_state = max_encoded_state - min_encoded_state
	scale_encoded_state = tf.where(scale_encoded_state<1e-5,scale_encoded_state+1e-5,scale_encoded_state)#avoid divided by 0 or too small value
	encoded_state_normalized = (to_scale_hidden_state - min_encoded_state) / scale_encoded_state
	if hidden_state_as_matrix:
		encoded_state_normalized=tf.reshape(encoded_state_normalized,shape)#and reshape to original shape
	return encoded_state_normalized
	
def support_to_scalar(logits, support_size, from_logits=True):# logits is in shape (batch_size,full_support_size)
	"""
	Transform a categorical representation to a scalar
	See paper appendix F Network Architecture (P.14)
	"""
	# Decode to a scalar
	if from_logits:
		probabilities=tf.nn.softmax(logits)
	else:
		probabilities=logits
	support=tf.range(-support_size,support_size+1,delta=1,dtype=tf.float32) # in shape (1,full_support_size)
	support=tf.expand_dims(support,axis=0)
	support=tf.tile(support,(probabilities.shape[0],1))
	x = tf.reduce_sum(support * probabilities, axis=1)

	# Invert the scaling (defined in https://arxiv.org/abs/1805.11593)
	x = tf.math.sign(x) * (
		((tf.math.sqrt(1 + 4 * 0.001 * (tf.math.abs(x) + 1 + 0.001)) - 1) / (2 * 0.001))
		** 2
		- 1
	)
	return x
def scalar_to_support(x, support_size):####todo: implement
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
	floor=tf.cast(floor,'int32')
	logits = tf.zeros((length * (2 * support_size + 1)))#flattened of (length , 2 * support_size + 1)
	ori_indices=floor+support_size
	indices_to_add=tf.range(length)*(2 * support_size + 1)
	indices=ori_indices+indices_to_add
	indices=tf.expand_dims(indices, axis=-1)# index is in 1-dimensional
	logits=tf.tensor_scatter_nd_update(
		logits,indices=indices,updates=1 - prob
	)
	ori_indices=ori_indices+1
	prob = tf.where(2 * support_size < ori_indices, 0., prob)
	ori_indices = tf.where(2 * support_size < ori_indices, 0, ori_indices)
	indices=ori_indices+indices_to_add
	indices=tf.expand_dims(indices, axis=-1)# index is in 1-dimensional
	logits=tf.tensor_scatter_nd_update(
		logits,indices=indices,updates=prob
	)
	logits=tf.reshape(logits,(*original_shape,-1))
	return logits

NetworkOutput=collections.namedtuple('NetworkOutput', ['reward', 'hidden_state','value','policy'])

												####shapes:###
#observation:		in shape of (batch_size,channels,board_size_x,board_size_y)#board_size_x=board_size_y in most cases
#channels=history_length*planes per image
#hidden_state:		in shape of (batch_size,hidden_state_size(32))
#					or (batch_size,hidden_state_size_x,hidden_state_size_y)
#action:			one-hotted, in shape of (batch_size,4+2*boardsize**2)
#policy:			in shape of (batch_size,4(UDLR))
#value and reward:	in shape of (batch_size,full_support_size) if using support, else (batch_size,1) #about "support", described at config.py:53
def my_adjust_dims(inputs,expected_shape_length):
	delta=len(inputs.shape)-expected_shape_length
	if delta<0:
		inputs=np.expand_dims(inputs,axis=list(range(-delta)))
	if delta>0:
		inputs=inputs.reshape((*inputs.shape[:-delta-1],-1))
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
	def representation(self,observation):
		#output:hidden_state
		pass
		
	@abstractmethod
	def dynamics(self, hidden_state, action):
		#output:hidden_state,reward
		pass
		
	@abstractmethod
	def prediction(self, hidden_state):
		#output:policy,value
		pass
	
	def initial_inference(self, observation)->NetworkOutput:
		'''
		directly inference, for training and reanalyse
		input shape: batch, channel, width, height
		'''
		assert len(observation.shape)==4
		hidden_state=scale_hidden_state(self.representation(observation))
		#hidden_state:batch, hidden_size
		policy,value=self.prediction(hidden_state)
		#policy:batch, 4
		#value:batch, 1 if not support, else batch, support*2+1
		return NetworkOutput(policy=policy,value=value,reward=None,hidden_state=hidden_state)
	def recurrent_inference(self, hidden_state, action):
		'''
		directly inference, for training
		'''
		hidden_state, reward=self.dynamics(hidden_state, action)
		hidden_state=scale_hidden_state(hidden_state)
		policy,value=self.prediction(hidden_state)
		return NetworkOutput(policy=policy,value=value,reward=reward,hidden_state=hidden_state)
	@abstractmethod
	def get_weights(self):
		pass
	@abstractmethod
	def set_weights(self,weights):
		pass
class Network:
	def __new__(cls,config):
		if config.network_type=="fullyconnected":
			return FullyConnectedNetwork(config)
		else:
			raise NotImplementedError
class FullyConnectedNetwork(AbstractNetwork):
	def __init__(self,config):
		super().__init__()
		#config
		self.config=config
		if config.support:
			self.full_support_size=2*config.support_size+1
		else:
			self.full_support_size=1
		self.support=config.support
		
		#network
		
		self.representation_model=self.one_output_model(
			config.observation_channels*config.board_size**2,
			config.representation_size,
			config.hidden_state_size)
		self.dynamics_model=self.two_outputs_model(
			config.hidden_state_size+4+2*config.board_size**2,
			config.dynamics_size,
			config.dynamics_hidden_state_head_size,
			config.dynamics_reward_head_size,
			config.hidden_state_size,
			self.full_support_size)#reward
		self.prediction_model=self.two_outputs_model(
			config.hidden_state_size,
			config.prediction_size,
			config.prediction_value_head_size,
			4,#policy
			self.full_support_size)#value
		self.trainable_variables=self.representation_model.trainable_variables+self.dynamics_model.trainable_variables+self.prediction_model.trainable_variables
	def representation(self, observation):
		observation=my_adjust_dims(observation, 2)
		return self.representation_model(observation)
	def dynamics(self, hidden_state, action):
		'''
		hidden state can be flattened or not
		'''
		hidden_state=my_adjust_dims(hidden_state, 2)
		concatenated_inputs=np.concatenate((hidden_state,action),axis=0)
		return self.dynamics_model(concatenated_inputs)
	def prediction(self, hidden_state):
		hidden_state=my_adjust_dims(hidden_state, 2)
		return self.prediction_model(hidden_state)
	class one_output_model(tf.keras.Model):
		def __init__(self,input_size,sizes,output_size):
			super().__init__()
			self.layers=[tf.keras.layers.Flatten()]
			for size in sizes+[output_size]:
				self.layers.append(tf.keras.layers.Dense(size,activation=tf.nn.relu))
		def call(self,x,training=False):
			for layer in self.layers:
				x=layer(x)
			return x
			
	class two_outputs_model(tf.keras.Model):
		def __init__(self,
				input_size,
				common_size,
				first_head_size,
				second_head_size,
				first_output_size,
				second_output_size):
			super().__init__()
			self.common_layers=[]
			self.first_head_layers=[]
			self.second_head_layers=[]
			
			for size in common_size:
				self.common_layers.append(tf.keras.layers.Dense(size,activation=tf.nn.relu))
				
			for size in first_head_size+[first_output_size]:
				self.first_head_layers.append(tf.keras.layers.Dense(size,activation=tf.nn.relu))
				
			for size in second_head_size+[second_output_size]:
				self.second_head_layers.append(tf.keras.layers.Dense(size,activation=tf.nn.relu))

		def call(self,x,training=False):
			for layer in self.common_layers:
				x=layer(x)
			
			first=dc(x)
			for layer in self.first_head_layers:
				first=layer(first)
			
			second=dc(x)
			for layer in self.second_head_layers:
				second=layer(second)
			return first,second
			'''
			first=first.numpy()
			second=second.numpy()
			return [(first[i],second[i]) for i in range(first.shape[0])]
			'''
			#return tf.concat((first,second),axis=1)
			
	def representation(self,observation):
		return scale_hidden_state(self.representation_model(observation))
	def dynamics(self,hidden_state,action):
		hidden_state,reward=self.dynamics_model(hidden_state,action)
		hidden_state=scale_hidden_state(hidden_state)
		return hidden_state,reward
	def prediction(self,hidden_state):
		policy,value=self.prediction_model(hidden_state)
		return policy,value
	def get_weights(self):
		return {'representation':self.representation_model.get_weights(),'dynamics':self.dynamics_model.get_weights(),'prediction':self.prediction_model.get_weights()}
	def set_weights(self,weights):
		'''
		set weights of 3 networks
		weights can be {name(str):weights(np.ndarray)}
		or list of 3 weights [weights(np.ndarray)]
		'''
		if isinstance(weights,dict):
			for name,weight in weights.items():
				if name=='representation':
					self.representation_model.set_weights(weight)
				elif name=='dynamics':
					self.dynamics_model.set_weights(weight)
				elif name=='prediction':
					self.prediction_model.set_weights(weight)
				else:
					raise NotImplementedError
		elif isinstance(weights,list):
			self.representation_model.set_weights(weights[0])
			self.dynamics_model.set_weights(weights[1])
			self.prediction_model.set_weights(weights[2])
		else:
			raise NotImplementedError
QueueItem=collections.namedtuple("QueueItem",['inputs','future'])
class Manager:
	'''
	Queuing requests of network prediction, and run them simultaneously to improve efficiency
	
	input to each network for a single prediction should be in [*expected_shape], rather than [batch_size(1),*expected_shape]
		process in self.prediction_worker
		and observation can be flattened or not
	'''
	def __init__(self, config, model:AbstractNetwork):
		self.support=config.support
		
		self.loop=asyncio.get_event_loop()
		#callable model
		self.model=model
		self.representation=model.representation
		self.dynamics=model.dynamics
		self.prediction=model.prediction
		self.representation_queue=asyncio.queues.Queue(config.model_max_threads)
		self.dynamics_queue=asyncio.queues.Queue(config.model_max_threads)
		self.prediction_queue=asyncio.queues.Queue(config.model_max_threads)
		
		self.coroutine_list=[self.prediction_worker()]
		async def push_queue_func(features,network:str):#network means which to use. If passing string consumes too much time, pass int instead.
			future=self.loop.create_future()
			item=QueueItem(features,future)
			if network=='representation':
				await self.representation_queue.put(item)
			elif network=='dynamics':
				await self.dynamics_queue.put(item)
			elif network=='prediction':
				await self.prediction_queue.put(item)
			else:
				raise NotImplementedError
			return future
		self.push_queue=push_queue_func
		
	def add_coroutine_list(self,toadd):
		#if toadd not in self.coroutine_list:
		self.coroutine_list.append(toadd)
			
	def run_coroutine_list(self):
		ret=self.loop.run_until_complete(asyncio.gather(*(self.coroutine_list)))
		self.coroutine_list=[self.prediction_worker()]
		return ret
	def are_all_queues_empty(self):
		for q in (self.representation_queue,self.dynamics_queue,self.prediction_queue):
			if not q.empty():
				return False
		return True
	def get_weights(self):
		return self.model.get_weights()
	def set_weights(self,weights):
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
			for name,queue,func in zip(('representation','dynamics','prediction'),(self.representation_queue,self.dynamics_queue,self.prediction_queue),(self.representation,self.dynamics,self.prediction)):
				if queue.empty():
					continue
				item_list = [queue.get_nowait() for _ in range(queue.qsize())]  # type: list[QueueItem]
				inputs=np.concatenate([np.expand_dims(item.inputs,axis=0) for item in item_list], axis=0)
				#print('nn',len(item_list))
				with tf.device('/device:GPU:0'):
					start=time()
					results = func(inputs)
					print('inference:',time()-start)
				if name=='representation':
					hidden_state=scale_hidden_state(results)#scale hidden state
					assert hidden_state.shape[0]==len(item_list), 'sizes of hidden_state('+hidden_state.shape+') and item_list('+len(item_list)+') don\'t match, this should never happen.'
					for i in range(len(item_list)):
						item_list[i].future.set_result(hidden_state[i])
				elif name=='dynamics':
					hidden_state,reward=results
					hidden_state=scale_hidden_state(hidden_state)#scale hidden state
					assert hidden_state.shape[0]==len(item_list) and reward.shape[0]==len(item_list), 'sizes of hidden_state('+hidden_state.shape+'), reward('+reward.shape+'), and item_list('+len(item_list)+') don\'t match, this should never happen.'
					if self.support:
						reward=support_to_scalar(reward,self.support)
					for i in range(len(item_list)):
						item_list[i].future.set_result((hidden_state[i],reward[i]))
				elif name=='prediction':
					policy,value=results
					assert policy.shape[0]==len(item_list) and value.shape[0]==len(item_list), 'sizes of policy('+policy.shape+'), value('+value.shape+'), and item_list('+len(item_list)+') don\'t match, this should never happen.'
					if self.support:
						value=support_to_scalar(value,self.support)
					for i in range(len(item_list)):
						item_list[i].future.set_result((policy[i],value[i]))
class Predictor:
	def __init__(self,manager):
		self.manager=manager
		self.push_queue=manager.push_queue
		
	async def get_outputs(self,inputs,network):
		future=await self.push_queue(inputs,network)
		await future
		return future.result()
		
	async def initial_inference(self,observation)->NetworkOutput:
		'''
		input shape:
			observation: whether flattened or not and without batchsize
			it will be flattened in tf.keras.layers.Flatten
		'''
		hidden_state=scale_hidden_state(await self.get_outputs(observation,'representation'))#scaled
		policy,value=await self.get_outputs(hidden_state,'prediction')
		return NetworkOutput(reward=0,hidden_state=hidden_state,value=value,policy=policy)

	async def recurrent_inference(self,hidden_state:np.array,action:np.array)->NetworkOutput:
		'''
		temporarily only for fully connected network.
		input shape:
			all be without batchsize
			hidden_state:(hidden_state_size(32)) or (hidden_state_size_x,hidden_state_size_y)
			action:one-hotted, (4+2*boardsize**2)
		'''
		if len(hidden_state.shape)==2:
			hidden_state.flatten()
		inputs=np.concatenate((hidden_state,action),axis=0)
		new_hidden_state,reward=await self.get_outputs(inputs,'dynamics')
		new_hidden_state=scale_hidden_state(new_hidden_state)#scaled
		
		policy,value=await self.get_outputs(new_hidden_state,'prediction')
		
		return NetworkOutput(reward=reward,hidden_state=new_hidden_state,value=value,policy=policy)

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model,layers,optimizers,losses
from time import time,sleep
import collections
from abc import ABC,abstractmethod

def scale_hidden_state(to_scale_hidden_state,hidden_state_as_matrix):
	shape=to_scale_hidden_state.shape
	if hidden_state_as_matrix:
		to_scale_hidden_state=tf.reshape(to_scale_hidden_state,(shape[0],-1))#flatten
	min_encoded_state = tf.math.reduce_min(to_scale_hidden_state,axis=1,keepdims=True)
	max_encoded_state = tf.math.reduce_max(to_scale_hidden_state,axis=1,keepdims=True)
	scale_encoded_state = max_encoded_state - min_encoded_state
	scale_encoded_state = tf.where(scale_encoded_state<1e-5,scale_encoded_state+1e-5,scale_encoded_state)#avoid divided by 0 or too small value
	encoded_state_normalized = (to_scale_hidden_state - min_encoded_state) / scale_encoded_state
	if hidden_state_as_matrix:
		encoded_state_normalized.reshape(shape)#and reshape to original shape
	return encoded_state_normalized

NetworkOutput=collections.namedtuple('NetworkOutput', ['reward', 'hidden_state','value','policy'])
#observation:		in shape of (batch_size,channels,board_size_x,board_size_y)#board_size_x=board_size_y in most cases
#channels=history_length*planes per image
#hidden_state:		in shape of (batch_size,hidden_state_size(32))
#					or (batch_size,hidden_state_size_x,hidden_state_size_y)
#action:			one-hotted, in shape of (batch_size,4+boardsize**2)
#policy:			in shape of (batch_size,4(UDLR))
#value and reward:	in shape of (batch_size,full_support_size) if using support, else (batch_size,1) #about "support", described at config.py:53
class AbstractNetwork(ABC):
	def __init__(self):
		super().__init__()
		pass
		
	@abstractmethod
	def representation(self,observation):
		#output:hidden_state
		pass
		
	@abstractmethod
	def dynamics(self,hidden_state,action):
		#output:hidden_state,reward
		pass
		
	@abstractmethod
	def prediction(self,hidden_state):
		#output:policy,value
		pass
	
	def initial_inference(self,observation)->NetworkOutput:
		hidden_state=self.representation(observation)#scaled
		policy,value=self.prediction(hidden_state)
		return NetworkOutput(reward=0,hidden_state=hidden_state,value=value,policy=policy)

	def recurrent_inference(self,hidden_state,action)->NetworkOutput:
		new_hidden_state,reward=self.dynamics(hidden_state,action)#scaled
		policy,value=self.prediction(new_hidden_state)
		return NetworkOutput(reward=reward,hidden_state=new_hidden_state,value=value,policy=policy)
		
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
			config.hidden_state_size+4,
			config.dynamics_size,
			config.dynamics_hidden_state_head_size,
			config.dynamics_reward_head_size,
			config.hidden_state_size,
			self.full_support_size)
		self.prediction_model=self.two_outputs_model(
			config.hidden_state_size,
			config.prediction_size,
			config.prediction_value_head_size,
			4,
			self.full_support_size)
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
			
			first=x
			for layer in self.first_head_layers:
				first=layer(first)
			
			second=x
			for layer in self.second_head_layers:
				second=layer(second)
			return first,second
			
	def representation(self,observation):
		return scale_hidden_state(self.representation_model(observation),False)
	def dynamics(self,hidden_state,action):
		hidden_state,reward=self.dynamics_model(hidden_state,action)
		hidden_state=scale_hidden_state(hidden_state)
		return hidden_state,reward
	def prediction(self,hidden_state):
		policy,value=self.prediction_model(hidden_size)
		return policy,value
QueueItem=collections.namedtuple("QueueItem",['inputs','future'])
class Manager:
	def __init__(self,representation_func,dynamics_func,prediction_func,max_threads=3000):
		self.loop=asyncio.get_event_loop()
		
		self.representation=representation_func
		self.dynamics=dynamics_func
		self.prediction=prediction_func
		
		self.representation_queue=asyncio.queues.Queue(max_threads)
		self.dynamics_queue=asyncio.queues.Queue(max_threads)
		self.prediction_queue=asyncio.queues.Queue(max_threads)
		
		self.coroutine_list=[self.prediction_worker()]
		async def push_queue(features,network->str):#network means which to use
			future=self.loop.create_future()
			item=QueueItem(inputs,future)
			if network=='representation'):
				await self.representation_queue.put(item)
			if network=='dynamics'):
				await self.dynamics_queue.put(item)
			if network=='prediction'):
				await self.prediction_queue.put(item)
			return future
		self.push_queue_func=push_queue
		
	def add_coroutine_list(self,toadd):
		if toadd not in self.coroutine_list:
			self.coroutine_list.append(toadd)
			
	def run_coroutine_list(self):
		ret=self.loop.run_until_complete(asyncio.gather(*(self.coroutine_list)))
		self.coroutine_list=[self.prediction_worker()]
		return ret
	async def prediction_worker(self):
		"""For better performance, queueing prediction requests and predict together in this worker.
		speed up about 3x.
		"""
		margin = 10  # avoid finishing before other searches starting.
		while margin > 0:
			if q.empty():
				await asyncio.sleep(1e-3)
				if q.empty() and margin > 0:
					margin -= 1
				continue
			for queue,func in zip((self.representation_queue,self.dynamics_queue,self.prediction_queue),(self.representation,self.dynamics,self.prediction)):
				item_list = [queue.get_nowait() for _ in range(queue.qsize())]  # type: list[QueueItem]
				inputs=np.concatenate([np.expand_dims(item.inputs,axis=0) for item in item_list])
				print('nn',len(item_list))
				with tf.device('/device:GPU:0'):
					#start=time()
					results = self.forward(features)
					#print('inference:',time()-start)
				for a,b,c,d,e,item in zip(*results,item_list):
					item.future.set_result((a,b,c,d,e))

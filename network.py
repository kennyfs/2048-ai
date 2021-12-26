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
		hidden_state=self.representation(observation)
		policy,value=self.prediction(hidden_state)
		return NetworkOutput(reward=0,hidden_state=hidden_state,value=value,policy=policy)

	def recurrent_inference(self,hidden_state,action)->NetworkOutput:
		new_hidden_state,reward=self.dynamics(hidden_state,action)
		policy,value=self.prediction(new_hidden_state)
		return NetworkOutput(reward=reward,hidden_state=new_hidden_state,value=value,policy=policy)
		
class FullyConnectedNetwork(AbstractNetwork):
	def __init__(self,config):
		super().__init__()
		#config
		self.action_space=4+config.board_size**2#UDLR and adding tiles
		if config.support:
			self.full_support_size=2*config.support_size+1
		#list meaning size of hidden layers
		self.representation_size=config.representation_size
		
		self.dynamics_size=config.dynamics_size
		self.dynamics_hidden_state_head_size=config.dynamics_hidden_state_head_size
		self.dynamics_reward_head_size=config.dynamics_reward_head_size
		
		self.prediction_size=config.prediction_size
		self.prediction_value_head_size=config.prediction_value_head_size
		self.prediction_policy_head_size=config.prediction_policy_head_size
		#network
		
	class representation_model(tf.keras.Model):
		def __init__(self,input_size,representation_size,output_size):
			super().__init__()
			self.layers=[]
			for size in representation_size+[output_size]:
				self.layers.append(tf.keras.layers.Dense(size,activation=tf.nn.relu))
		def call(self,x,training=False):
			for layer in self.layers:
				x=layer(x)
			return x
			
	class dynamics_model(tf.keras.Model):
		def __init__(self,
				input_size,
				dynamics_size,
				dynamics_hidden_state_head_size,
				dynamics_reward_head_size,
				hidden_state_size,
				reward_size):
			super().__init__()
			self.common_layers=[]
			self.hidden_state_head_layers=[]
			self.reward_head_layers=[]
			
			for size in dynamics_size:
				self.common_layers.append(tf.keras.layers.Dense(size,activation=tf.nn.relu))
				
			for size in dynamics_hidden_state_head_size+[hidden_state_size]:
				self.hidden_state_head_layers.append(tf.keras.layers.Dense(size,activation=tf.nn.relu))
				
			for size in dynamics_reward_head_size+[reward_size]:
				self.reward_head_layers.append(tf.keras.layers.Dense(size,activation=tf.nn.relu))
		def call(self,x,training=False):
			for layer in self.common_layers:
				x=layer(x)
			
			hidden_state=x
			for layer in self.hidden_state_hard_layers:
				hidden_state=layer(hidden_state)
			
			reward=x
			for layer in self.reward_head_layers:
				reward=layer(reward)
			return hidden_state,reward
			
	def representation(self,observation):
		return scale_hidden_state(self.representation_model(observation),False)
	def dynamics(self,hidden_state,action):
		hidden_state,reward=self.dynamics_network(hidden_state,action)
		

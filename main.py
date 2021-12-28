from random import randint,random
from copy import copy,deepcopy
#from network import nn
import numpy as np
from time import time,sleep
from environment import Environment,Action
from config import Config,Game,default_config
from network import *
import collections
bg   ="\x1b[48;5;"
word ="\x1b[38;5;"
end  ="m"
reset="\x1b[0m"
'''
assumption:
total training steps=1e5 or less
'''
def get_features(grid):#given grid, return features for Network input
	sz=len(grid)
	grid=np.array(grid)
	result=[]
	for i in range(1,sz*sz+1):
		result.append(np.where(grid==i,1.0,0.0))
	return np.array(result)
	  
MAXIMUM_FLOAT_VALUE=float('inf')
KnownBounds = collections.namedtuple('KnownBounds', ['min', 'max'])
class MinMaxStats():
	"""A class that holds the min-max values of the tree."""

	def __init__(self,known_bounds:KnownBounds):
		self.maximum=known_bounds.max if known_bounds else -MAXIMUM_FLOAT_VALUE
		self.minimum=known_bounds.min if known_bounds else MAXIMUM_FLOAT_VALUE
		#any update is accepted
		
	def update(self,value:float):
		self.maximum=max(self.maximum,value)
		self.minimum=min(self.minimum,value)

	def normalize(self,value:float)->float:
		if self.maximum>self.minimum:
			# We normalize only when we have set the maximum and minimum values.
			return (value-self.minimum)/(self.maximum-self.minimum)
		return value

class ActionHistory():
	"""Simple history container used inside the search.

	Only used to keep track of the actions executed.
	"""

	def __init__(self, history:'List[Action]', action_space_size: int):
		self.history = list(history)
		self.action_space_size = action_space_size

	def clone(self):
		return ActionHistory(self.history, self.action_space_size)

	def add_action(self, action: Action):
		self.history.append(action)

	def last_action(self) -> Action:
		return self.history[-1]

	def action_space(self) -> 'List[Action]':
		return [Action(i) for i in range(self.action_space_size)]

class Node():
	def __init__(self,p:float):
		self.visit_count=0
		self.p=p
		self.value_sum=0
		self.children={}#Dict[Action,Node]
		self.hidden_state=None
		self.reward=0

	def expanded(self)->bool:
		return len(self.children)>0

	def value(self)->float:
		if self.visit_count==0:
			return 0
		return self.value_sum/self.visit_count

class ReplayBuffer():### Starting from here next time
	def __init__(self, config: Config):
		self.window_size = config.window_size
		self.batch_size = config.batch_size
		self.buffer = []
		#In werner-duvaud/muzero-general, buffer is dict, which keys are increasing index, and values are corresponding games.
		#In this way, update priority(described in Muzero P.15 Appendix G Training) and update game history are easier to implement
		#I haven't decided how to store buffer now.
		#Maybe I won't use priority, therefore no need for using dict.
	def save_game(self, game):
		if len(self.buffer) > self.window_size:
			self.buffer.pop(0)
		self.buffer.append(game)
		
	def sample_batch(self, num_unroll_steps: int, td_steps: int):
		games=self.sample_n_games(self.batch_size)
		game_pos=[(g,self.sample_position(g))for g in games]
		return [(g.make_history_images(i),g.history[i:i+num_unroll_steps],
						 g.make_target(i,num_unroll_steps))
						for (g, i) in game_pos]

	def sample_n_games(self,n_games) -> 'np.ndarray[Game]':
		# Sample game from buffer either uniformly or according to some priority.
		return np.random.choice(self.buffer, n_games)

	def sample_position(self, game) -> int:
		# Sample position from game either uniformly or according to some priority.
		return np.random.choice(len(game_history.root_values))
		

class SharedStorage():

	def __init__(self):
		self._networks={}#{train_step(int):Network}

	def latest_network(self):#return a network(with representation, dynamics, prediction)
		if self._networks:
			return self._networks[max(self._networks.keys())]
		else:
			#policy->uniform,value->0,reward->0
			return make_network()

	def save_network(self,step:int,network):
		self._networks[step]=network
	
def player():
	config=default_config()
	config.board_size=10
	b=Environment(config)
	b.add()
	b.add()
	l=0
	while not b.finish():
		b.dump()
		print('move count:',l)
		l+=1
		done=False
		while not done:
			valid=False
			while not valid:
				valid=True
				try:a=int(input())
				except:valid=False
				if a>3 or a<0:
					valid=False
			if b.valid(Action(a)):
				b.step(Action(a))
				b.add()
				done=True
	b.dump()
player()

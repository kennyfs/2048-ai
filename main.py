from random import randint,random
from copy import copy,deepcopy
#from network import nn
import numpy as np
from time import time,sleep
from environment import Environment
bg   ="\x1b[48;5;"
word ="\x1b[38;5;"
end  ="m"
reset="\x1b[0m"
'''
assumption:
total training steps=1e5 or less
'''
def default_visit_softmax_temperature(num_moves,training_steps):
	if training_steps < 50e3:
	  return 1.0
	elif training_steps < 75e3:
	  return 0.5
	else:
	  return 0.25
class Config:
	def __init__(self,
				action_space_size:int,
				max_moves:int,
				discount:float,
				dirichlet_alpha:float,
				num_simulations:int,
				batch_size:int,
				td_steps:int,#when calculating value target, bootstrapping td_steps steps next moves' rewards and value
				num_actors:int,
				lr_init:float,
				lr_decay_steps:float,
				visit_softmax_temperature_fn,
				known_bounds:Optional[KnownBounds]=None):
	### Self-Play
	self.action_space_size=action_space_size
	self.num_actors=num_actors

	self.visit_softmax_temperature_fn=visit_softmax_temperature_fn
	self.max_moves=max_moves
	self.num_simulations=num_simulations
	self.discount=discount

	# Root prior exploration noise.
	self.root_dirichlet_alpha=dirichlet_alpha
	self.root_exploration_fraction=0.25

	# UCB formula
	self.pb_c_base=19652
	self.pb_c_init=1.25

	# If we already have some information about which values occur in the
	# environment, we can use them to initialize the rescaling.
	# This is not strictly necessary, but establishes identical behaviour to
	# AlphaZero in board games.
	self.known_bounds=known_bounds

	###Training
	self.training_steps=int(100e3)
	self.checkpoint_interval=int(5e2)
	self.window_size=int(1e6)
	self.batch_size=batch_size
	self.num_unroll_steps=5
	self.td_steps=td_steps

	self.weight_decay=1e-4
	self.momentum=0.9

	#Exponentiallearningrateschedule
	self.lr_init=lr_init
	self.lr_decay_rate=0.1
	self.lr_decay_steps=lr_decay_steps

	def new_game(self):
		return Game(self.action_space_size,self.discount)
def default_config():
	return Config(action_space_size=4,
				max_moves=1e5,#it can be infinity because any 2048 game is bound to end
				discount=0.9,
				dirichlet_alpha=0.3,
				num_simulations=100,
				batch_size=1024,
				td_steps=15,#when calculating value target, bootstrapping td_steps steps next moves' rewards and value
				num_actors=1000,
				lr_init=0.1,
				lr_decay_steps=35e3,
				visit_softmax_temperature_fn=default_visit_softmax_temperature)
class Action:
	#0~3:up,down,left,right
	#starting from 4:put a tile, 4~19 for putting a 2, 20~35 for putting a 4
	#(treat putting a tile as an action)
	def __init__(self,index:int):
		self.index = index
	def __hash__(self):
		return self.index
	def __eq__(self, other):
		return self.index == other.index
	def __gt__(self, other):
		return self.index > other.index
class Game:
	# Game is not responsible for record game
	def __init__(self,config:Config):
		self.environment=Environment()  # Game specific environment.
		self.history=[]
		self.child_visits=[]
		self.root_values=[]
		self.action_space_size=config.action_space_size
		self.discount=config.discount
	def terminal(self)->bool:
		# if the game ends
		return self.environment.finish()
	def legal_actions(self)->List[Action]:
		# list of legal actions
		return self.environment.legal_actions()

	def apply(self,action:Action):
		reward = self.environment.step(action)
		self.rewards.append(reward)
		self.history.append(action)

	def store_search_statistics(self, root: Node):
		sum_visits = sum(child.visit_count for child in root.children.values())
		action_space = (Action(index) for index in range(self.action_space_size))
		self.child_visits.append([
				root.children[a].visit_count / sum_visits if a in root.children else 0
				for a in action_space
		])
		self.root_values.append(root.value())

	def make_image(self, state_index: int):
		# Game specific feature planes.
		return []

	def make_target(self, state_index: int, num_unroll_steps: int, td_steps: int,
									to_play: Player):
		# The value target is the discounted root value of the search tree N steps
		# into the future, plus the discounted sum of all rewards until then.
		targets = []
		for current_index in range(state_index, state_index + num_unroll_steps + 1):
			bootstrap_index = current_index + td_steps
			if bootstrap_index < len(self.root_values):
				value = self.root_values[bootstrap_index] * self.discount**td_steps
			else:
				value = 0

			for i, reward in enumerate(self.rewards[current_index:bootstrap_index]):
				value += reward * self.discount**i	# pytype: disable=unsupported-operands

			# For simplicity the network always predicts the most recently received
			# reward, even for the initial representation network where we already
			# know this reward.
			if current_index > 0 and current_index <= len(self.rewards):
				last_reward = self.rewards[current_index - 1]
			else:
				last_reward = 0

			if current_index < len(self.root_values):
				targets.append((value, last_reward, self.child_visits[current_index]))
			else:
				# States past the end of games are treated as absorbing states.
				targets.append((0, last_reward, []))
		return targets

	def to_play(self) -> Player:
		return Player()

	def action_history(self) -> ActionHistory:
		return ActionHistory(self.history, self.action_space_size)
	def addrand(self,x,y,v):
		self.history.append((2,x,y,v))
	def step(self,action):
		self.history.append((1,action))
	def replay(self):
		tmp=Environment([[0]*4 for i in range(4)],addwhenplay=False)
		for i in self.history:
			sleep(0.5)
			if i[0]==1:
				tmp.step(i[1])
			else:
				tmp.grid[i[1]][i[2]]=i[3]
			tmp.dump()
			sleep(0.5)
def player():
	g=Game()
	b=Environment(g=g)
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
			if b.valid(a):
				b.step(a)
				x,y,v=b.add()
				g.addrand(x,y,v)
				done=True
	b.dump()
	g.write('test.sgf')
player()

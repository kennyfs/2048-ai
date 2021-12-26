def default_visit_softmax_temperature(num_moves,training_steps):
	if training_steps < 50e3:
	  return 1.0
	elif training_steps < 75e3:
	  return 0.5
	else:
	  return 0.25

class Config:
	def __init__(self,
				action_space_type0_size:int,
				action_space_type1_size:int,
				max_moves:int,
				discount:float,
				dirichlet_alpha:float,
				num_simulations:int,
				board_size:int,
				batch_size:int,
				td_steps:int,#when calculating value target, bootstrapping td_steps steps next moves' rewards and value
				num_actors:int,#how many games run at same time
				lr_init:float,
				lr_decay_steps:float,
				visit_softmax_temperature_fn,
				known_bounds=None):
		### Self-Play
		self.action_space_type0_size=action_space_type0_size
		self.action_space_type1_size=action_space_type1_size
		self.num_actors=num_actors
		self.board_size=board_size

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
		### Network info
		self.hidden_state_size=32
		self.network_type='fullyconnected'#'resnet'/'fullyconnected'/...
		self.support=False# whether to use support (using an array to represent reward and value(discounted), e.g. 3.7=3*0.3+4*0.7, so [0,0,0,0.3,0.7,0...])
		
		# Fully Connected Network
		self.encoding_size = 32
		self.fc_representation_layers = []  # Define the hidden layers in the representation network
		self.fc_dynamics_layers = [64]  # Define the hidden layers in the dynamics network
		self.fc_reward_layers = [64]  # Define the hidden layers in the reward network
		self.fc_value_layers = []  # Define the hidden layers in the value network
		self.fc_policy_layers = []
		
		### Training
		self.training_steps=int(100e3)
		self.checkpoint_interval=int(5e2)
		self.window_size=int(1e6)#max game cnt stored in replaybuffer
		self.batch_size=batch_size
		self.num_unroll_steps=5
		self.td_steps=td_steps

		self.weight_decay=1e-4
		self.momentum=0.9

		#Exponentiallearningrateschedule
		self.lr_init=lr_init
		self.lr_decay_rate=0.1
		self.lr_decay_steps=lr_decay_steps

class Game:
	# In any game, the first two actions are adding tile, then each move is followed by adding tile
	def __init__(self,config):
		self.environment=Environment(config)  # Game specific environment.
		self.history=[]#List[Action]
		self.type=[]#0:move,1:adding tile # will also stored in Node
		self.child_visits=[]
		self.root_values=[]
		self.action_space_type0_size=config.action_space_type0_size
		self.action_space_type1_size=config.action_space_type1_size
		self.discount=config.discount
		
	def terminal(self)->bool:
		# if the game ends
		return self.environment.finish()
		
	def legal_actions(self)->'List[Action]':
		# list of legal actions, only care about move
		return self.environment.legal_actions()
		
	def apply(self,action):
		reward = self.environment.step(action)
		self.rewards.append(reward)
		self.history.append(action)
		self.type.append(1 if 0<=action.type and action.type<=3 else 0)
		
	def store_search_statistics(self, root):
		#only store type 0(move)
		#root.children is Dict[Node] whose keys are actions
		sum_visits = sum(child.visit_count for child in root.children.values())
		action_space = (Action(index) for index in range(self.action_space_type0_size))
		self.child_visits.append([
				root.children[a].visit_count / sum_visits if a in root.children else 0
				for a in action_space
		])
		self.root_values.append(root.value())
		#adding tile
		self.child_visits.append(None)
		self.root_values.append(None)
		
	def make_image(self, state_index: int):#-1 means the last
		# Game specific feature planes.
		# Go through history
		tmpenv=Environment()
		if state_index==-1:
			for action in self.history:
				tmpenv.step(action)
		elif 0<=state_index and state_index<=len(self.history):
			# do history[:state_index]
			for action in self.history[:state_index]:
				tmpenv.step(action)
		else:
			raise BaseException('state_index('+str(state_index)+') is out of range('+str(len(self.history))+') (in Game.make_image)')
		return get_features(tmpenv.grid)

	def make_target(self, state_index: int, num_unroll_steps: int, td_steps: int):#return type List[Tuple(value_target,reward_target,policy_target) policy might not have a sum of 1
		# The value target is the discounted root value of the search tree N steps
		# into the future, plus the discounted sum of all rewards until then.
		targets = []
		for current_index in range(state_index, state_index + num_unroll_steps + 1):
			bootstrap_index = current_index + td_steps
			if bootstrap_index>len(self.root_values)-1:
				break
			if self.type[bootstrap_index]==1:
				continue#only makes target for type0(move)
			value = self.root_values[bootstrap_index] * self.discount**td_steps
			dis=1
			for reward in self.rewards[current_index:bootstrap_index]:
				value += reward * dis	# pytype: disable=unsupported-operands
				dis*=self.discount

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
		
	def action_history(self):
		return ActionHistory(self.history, self.action_space_type0_size)
		
	def replay(self):
		tmp=Environment([[0]*4 for i in range(4)],addwhenplay=False)
		for i in self.history:
			sleep(0.8)
			tmp.step(i)
			tmp.dump()
			
	def write(self,f):
		with open(f,'w') as F:
			F.write(str(self.data))
			F.write('\n')
			F.write(str(self.moves))
			
	def read(self,f):#clear origin data, moves
		with open(f,'r') as F:
			self.data=eval(F.readline())
			self.moves=eval(F.readline())

def default_config():
	return Config(
		action_space_type0_size=4,
		action_space_type1_size=32,
		max_moves=1e5,#it can be infinity because any 2048 game is bound to end
		discount=0.97,
		dirichlet_alpha=0.3,
		num_simulations=100,
		board_size=4,
		batch_size=1024,
		td_steps=10,#when calculating value target, bootstrapping td_steps steps next moves' rewards and value
		num_actors=1000,
		lr_init=0.1,
		lr_decay_steps=35e3,
		visit_softmax_temperature_fn=default_visit_softmax_temperature)

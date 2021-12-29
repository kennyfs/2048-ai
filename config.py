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
		self.action_space_type0_size=4
		self.action_space_type1_size=board_size**2
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
		self.observation_channels=self.board_size**2
		self.network_type='fullyconnected'#'resnet'/'fullyconnected'/...
		self.support=0# the size of support (using an array to represent reward and value(discounted), e.g. 3.7=3*0.3+4*0.7, so [0,0,0,0.3,0.7,0...])
		#this = 0 means not using support
		
		# Fully Connected Network
		self.hidden_state_size=32
		self.representation_size=[32,32,32]# Define the hidden layers in the representation network
		self.dynamics_size=[32,32]# Define the hidden layers in the common parts of the dynamics network
		self.dynamics_hidden_state_head_size=[32]# Define the hidden layers in hidden state head of the dynamics network
		self.dynamics_reward_head_size=[32]# Define the hidden layers in reward head of the dynamics network
		self.prediction_size=[32,32]# Define the hidden layers in the common parts of the prediction network
		self.prediction_value_head_size=[32]# Define the hidden layers in value head of the prediction network
		self.prediction_policy_head_size=[32]# Define the hidden layers in policy head of the prediction network
		
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

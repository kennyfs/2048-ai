import numpy as np
import os
import datetime

def default_visit_softmax_temperature(num_moves=0,training_steps=0):
	if training_steps < 50e3:
		return 1.0
	elif training_steps < 75e3:
		return 0.5
	else:
	 	return 0.25

class Config:
	def __init__(self,
				max_moves:int,
				discount:float,
				search_threads:int,
				model_max_threads:int,
				if_add_exploration_noise:bool,
				dirichlet_alpha:float,
				num_simulations:int,
				board_size:int,
				batch_size:int,
				td_steps:int,#when calculating value target, bootstrapping td_steps steps next moves' rewards and value
				num_actors:int,#how many games run at same time
				lr_init:float,
				lr_decay_steps:float,
				visit_softmax_temperature_fn,
				known_bounds=None,
				save_game_to_file=True):
		### Self-Play
		self.action_space_type0=list(range(4))
		self.action_space_type1=list(range(4,4+2*board_size**2))
		self.type1_p=np.array([0]*4+[9]*board_size**2+[1]*board_size**2,dtype='float32')
		self.type1_p/=np.sum(self.type1_p)#used in MCTS 
		self.num_actors=num_actors
		self.board_size=board_size

		self.visit_softmax_temperature_fn=visit_softmax_temperature_fn
		self.max_moves=max_moves
		self.num_simulations=num_simulations
		self.discount=discount
		self.search_threads=search_threads
		self.model_max_threads=model_max_threads
		# Root prior exploration noise.
		self.if_add_exploration_noise=if_add_exploration_noise
		self.root_dirichlet_alpha=dirichlet_alpha
		self.root_exploration_fraction=0.25
		self.observation_shape=(self.board_size**2,self.board_size,self.board_size)
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
		self.support=100# the size of support (using an array to represent reward and value(discounted), e.g. 3.7=3*0.3+4*0.7, so [0,0,0,0.3,0.7,0...])
		#this = 0 means not using support
		
		# Fully Connected Network
		self.hidden_state_size=64
		self.representation_size=[128,64,32]# Define the hidden layers in the representation network
		self.dynamics_size=[128,128,128]# Define the hidden layers in the common parts of the dynamics network
		self.dynamics_hidden_state_head_size=[128,128]# Define the hidden layers in hidden state head of the dynamics network
		self.dynamics_reward_head_size=[64,64]# Define the hidden layers in reward head of the dynamics network
		self.prediction_size=[128,84,64]# Define the hidden layers in the common parts of the prediction network
		self.prediction_value_head_size=[64,64]# Define the hidden layers in value head of the prediction network
		self.prediction_policy_head_size=[64,64]# Define the hidden layers in policy head of the prediction network
		#Fully Connected Network doesn't work well

		### Training
		self.results_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "results", datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S"))
		self.training_steps=int(100e3)
		self.checkpoint_interval=int(5e2)
		self.window_size=int(1e6)#max game cnt stored in replaybuffer
		self.batch_size=batch_size
		self.num_unroll_steps=10
		self.optimizer='SGD'
		self.value_loss_weight=0.5#See paper appendix H Reanalyze
		self.steps_per_batch=10
		self.save_model=True
		#count adding(type 1), but not count them as network training target
		self.td_steps=td_steps

		self.weight_decay=1e-4
		self.momentum=0.9

		#Exponential learning rate schedule
		self.learning_rate_init=lr_init
		self.learning_rate_decay_rate=0.9
		self.learning_rate_decay_steps=lr_decay_steps

		self.save_game_to_file=save_game_to_file
		self.seed=None

		self.PER=True
		self.PER_alpha=1

		self.debug=False
		self.reanalyze=True#Using latest model's predcition for value to improve quality of value target(Appendix H)

		#log
		self.test_delay=0
		self.log_delay=2.0


		self.replay_buffer_size=1000
		#overall hyperparameter
		self.training_steps_to_selfplay_steps_ratio=0.2
		self.reanalyze_games_to_selfplay_games_ratio=0.8
		self.selfplay_games_to_test_games_ratio=0.1
		self.selfplay_games_per_run=10
def default_config():
	return Config(
		max_moves=1000000,#it can be infinity because any 2048 game is bound to end
		discount=0.97,
		search_threads=5,
		model_max_threads=3000,
		if_add_exploration_noise=True,
		dirichlet_alpha=0.3,
		num_simulations=100,
		board_size=4,
		batch_size=1024,
		td_steps=30,#when calculating value target, bootstrapping td_steps steps next moves' rewards and value
		#2048 games tend to be very long
		num_actors=5,
		lr_init=0.01,
		lr_decay_steps=35e3,
		visit_softmax_temperature_fn=default_visit_softmax_temperature)

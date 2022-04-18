import numpy as np
import os
import datetime

def default_visit_softmax_temperature(num_moves = 0, training_steps = 0):
	if training_steps < 50e3:
		return 0.5
	elif training_steps < 75e3:
		return 0.3
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
				td_steps:int, #when calculating value target, bootstrapping td_steps steps next moves' rewards and value
				num_actors:int, #how many games run at same time
				lr_init:float,
				lr_decay_steps:float,
				visit_softmax_temperature_fn,
				known_bounds = None,
				save_game_to_file = True):
		###Game imformation
		self.board_size = board_size
		self.action_space_type0 = list(range(4))
		self.action_space_type1 = list(range(4, 4+2*board_size**2))
		self.type1_p = np.array([0]*4+[9]*board_size**2+[1]*board_size**2, dtype = np.float32)
		self.type1_p /= np.sum(self.type1_p)#used in MCTS initializing type 1 nodes' policy
		### Self-Play
		self.num_actors = num_actors
		self.save_game_to_file = save_game_to_file
		#not used now, as parallel selfplay is too hard to implement with queuing manager
		self.max_moves = max_moves#selfplay games are forced to terminate after max_moves
		# MCTS
		self.visit_softmax_temperature_fn = visit_softmax_temperature_fn
		self.num_simulations = num_simulations#MCTS search times
		self.discount = discount#used in MCTS backpropagate and replaybuffer target value
		self.search_threads = search_threads
		self.model_max_threads = model_max_threads#for manager
		# Root prior exploration noise.
		self.if_add_exploration_noise = if_add_exploration_noise
		self.root_dirichlet_alpha = dirichlet_alpha
		self.root_exploration_fraction = 0.25
		# UCB formula
		self.pb_c_base = 19652
		self.pb_c_init = 1.25

		# If we already have some information about which values occur in the
		# environment, we can use them to initialize the rescaling.
		# This is not strictly necessary, but establishes identical behaviour to
		# AlphaZero in board games.
		self.known_bounds = known_bounds
		### Network info
		self.observation_shape = [self.board_size**2, self.board_size, self.board_size]
		self.network_type = 'resnet'#'resnet'/'fullyconnected'/...
		self.support = 50# the size of support (using an array to represent reward and value(discounted), e.g. 3.7 = 3*0.3+4*0.7, so [0, 0, 0, 0.3, 0.7, 0...])
		#this = 0 means not using support. Muzero uses 300, which can represent up to 90000.
		
		# Fully Connected Network
		self.hidden_state_size = 128
		self.representation_size = [128]*5# Define the hidden layers in the representation network
		self.dynamics_size = [128]*5# Define the hidden layers in the common parts of the dynamics network
		self.dynamics_hidden_state_head_size = []# Define the hidden layers in hidden state head of the dynamics network
		self.dynamics_reward_head_size = [128]# Define the hidden layers in reward head of the dynamics network
		self.prediction_size = []# Define the hidden layers in the common parts of the prediction network
		self.prediction_value_head_size = [128]# Define the hidden layers in value head of the prediction network
		self.prediction_policy_head_size = []# Define the hidden layers in policy head of the prediction network
		#Fully Connected Network doesn't work well

		#ResNet Network
		self.num_channels = 128
		self.num_blocks = 3
		self.num_blocks_dynamic_1 = 2
		self.num_blocks_dynamic_2 = 2
		#smaller network doesn't work well, all value output is about 170~200 and reward outputs are about 6
		self.reduced_channels_value = 4#conv1x1 planes following hidden_state
		self.reduced_channels_policy = 1
		self.reduced_channels_reward = 4
		self.value_layers = [128]# dense layer sizes following conv1x1 and flatten
		self.policy_layers = []
		self.reward_layers = [128]
		
		self.chance_filters = [32, 8] # times 1/4 per layer
		self.chance_output = 1
		### Training
		self.results_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "results", datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S"))
		if not os.path.isdir(self.results_path):
			os.mkdir(self.results_path)
		
		self.load_game_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "saved_games v4", 'resnet')
		self.save_game_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "saved_games v4", 'resnet')
		assert os.path.isdir(self.load_game_dir)
		assert os.path.isdir(self.save_game_dir)
		#self.save_game_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "saved_games v2", 'random')
		#change dir if you want
		#haven't change game loading path in replay_buffer.py
		self.training_steps = int(100e3)
		self.checkpoint_interval = int(1e3)
		self.window_size = int(1e6)#max game cnt stored in replaybuffer
		self.batch_size = batch_size
		self.num_unroll_steps = 5#for each gamepos chosen to be collected, go forward num_unroll_steps steps(for training dynamics)
		self.td_steps = td_steps
		self.optimizer = 'Adam'
		self.momentum = 0.9
		self.loss_weights = [1, 4, 4]#See paper appendix H Reanalyze
		s = sum(self.loss_weights)
		for i in range(3):
			self.loss_weights[i] /= s
		self.l2_weight = 1e-3 if self.optimizer == 'SGD' else 1e-4
		#value reward policy
		self.training_steps_per_batch = 10
		self.save_model = True
		#self.weight_decay = 1e-4 #useless for now
		self.discount_to_n = [self.discount**i for i in range(self.td_steps+5)]

		#Exponential learning rate schedule
		self.learning_rate_init = lr_init
		self.learning_rate_decay_rate = 1.0#more intuitive to use 0.5
		self.learning_rate_decay_steps = lr_decay_steps

		self.seed = None

		self.PER = False
		self.PER_alpha = 1

		self.debug = False
		self.reanalyze = True#Using latest model's predcition for value to improve quality of value target(Appendix H)

		#log(not used)
		#self.test_delay = 0
		#self.log_delay = 2.0


		self.replay_buffer_size = int(1e6)

		#overall hyperparameters
		self.training_steps_to_selfplay_steps_ratio = 1.0
		#don't need to be very high(e.g. 1, 1.2)
		#self.training_steps_to_selfplay_steps_ratio = float('inf')#observing training
		self.reanalyze_games_to_selfplay_games_ratio = 0.1#ratio to control the number of reanalyze games each time
		#reanalyze all games, as it doesn't take much time.
		self.test_games_to_selfplay_games_ratio = 0.1
		self.num_selfplay_game_per_iteration = 1
		self.num_random_games_per_iteration = 10000
		#manager config
		self.manager_queue = True

		#experiment
		self.winer_takes_all = False

		self.rotate = True
		self.flip = True
def default_config():
	return Config(
		max_moves = 1000000, #it can be infinity because any 2048 game is bound to end
		discount = 1.0,
		search_threads = 5,
		model_max_threads = 3000,
		if_add_exploration_noise = True,
		dirichlet_alpha = 0.3,
		num_simulations = 50,
		board_size = 4,
		batch_size = 512, #1024
		td_steps = 1, #when calculating value target, bootstrapping td_steps steps next moves' rewards and value
		#2048 games tend to be very long
		num_actors = 5,
		#lr_init = 1., #for adadelta
		lr_init = 8e-4, #1e-3
		lr_decay_steps = 40e3,
		visit_softmax_temperature_fn = default_visit_softmax_temperature)
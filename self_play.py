import asyncio
import collections
import copy
import math
import pickle
import environment
import random
import sys

import numpy as np
import tensorflow as tf
import network
import my_config
MAXIMUM_FLOAT_VALUE=float('inf')
KnownBounds = collections.namedtuple('KnownBounds', ['min', 'max'])
class MinMaxStats():
	"""A class that holds the min-max values of the tree."""

	def __init__(self,known_bounds:KnownBounds=None):
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
class GameHistory:
	"""
	Store only usefull information of a self-play game.
	
	IMPORTANT!!!
	at time t, gamehistory[t] should contain the information about:
	observation, value, child_visits at time t
	action, reward, action_type arising from t-1 to t
	"""
	#states followed by actions of type x = "type x state"
	def __init__(self):
		self.observation_history = []
		self.child_visits = []
		self.root_values = []

		self.initial_add = []
		self.action_history = []
		self.reward_history = []

		self.length = 0#+1 when calling GameHistory.add
		self.reanalyzed_predicted_root_values = None
		# For PER
		self.priorities = None
		self.game_priority = None

	def store_search_statistics(self, root, action_space, winer_takes_all=False):
		'''
		Turn visit count from root into a policy, store policy and valuesss
		'''
		if root is not None:
			sum_visits = sum(child.visit_count for child in root.children.values())
			if winer_takes_all:
				visits=[root.children[a].visit_count
						if a in root.children
						else 0
						for a in action_space]
				mx=max(visits)
				s=0
				for v in visits:
					if v==mx:
						s+=1
				self.child_visits.append([1/s if v==mx else 0 for v in visits])
			else:
				self.child_visits.append(
					[
						root.children[a].visit_count / sum_visits
						if a in root.children
						else 0
						for a in action_space
					]
				)

			self.root_values.append(root.value())
		else:
			raise BaseException('store_search_statistics receive root as None')

	def get_stacked_observations(self, index, num_stacked_observations):
		"""
		Generate a new observation with the observation at the index position
		and num_stacked_observations past observations and actions stacked
		according to gamehistory.
		"""
		# Convert to positive index
		index = index % self.length

		stacked_observations = self.observation_history[index].copy()
		for past_observation_index in reversed(
			range(index - num_stacked_observations, index)
		):
			if 0 <= past_observation_index:
				previous_observation = np.concatenate(
					(
						self.observation_history[past_observation_index],
						[
							np.ones_like(stacked_observations[0])
							* self.action_history[past_observation_index + 1]
						],
					)
				)
			else:
				previous_observation = np.concatenate(
					(
						np.zeros_like(self.observation_history[index]),
						[np.zeros_like(stacked_observations[0])],
					)
				)

			stacked_observations = np.concatenate(
				(stacked_observations, previous_observation)
			)

		return stacked_observations
	def get_observation(self, index):
		index = index % self.length
		return self.observation_history[index].copy()
	def save(self, file):
		with open(file,'w') as F:
			F.write(f'{self.initial_add[0]}\n')
			F.write(f'{self.initial_add[1]}\n')
			for action,reward,visits,value in zip(self.action_history,self.reward_history,self.child_visits,self.root_values):
				F.write(f'{action} {reward} {visits} {value}\n')
	def load(self, file, config, predictor=None):
		#about 60 times per second
		env=environment.Environment(config)
		with open(file,'r') as F:
			for _ in range(2):#initial 2 add actions
				env.step(eval(F.readline()[:-1]))
			for line in F.readlines():
				last_index=0
				assert(line[0]=='[')
				index=line.find(']',last_index)+1
				actions=eval(line[last_index:index])
				self.action_history.append(actions)
				reward=env.step(actions[0])
				self.observation_history.append(env.get_features())

				last_index=index+1
				index=line.find(' ',last_index)#[last_index,index) is reward
				self.reward_history.append(reward)

				if predictor:#means debug
					env.render()
					predictor.manager.add_coroutine_list(predictor.initial_inference(self.observation_history[-1]))
					if len(self.observation_history)>=2:
						predictor.manager.add_coroutine_list(predictor.recurrent_inference(self.observation_history[-2],self.action_history[-1]))
					out=predictor.manager.run_coroutine_list(True)
					output=out[0]
					print(f'value:{output.value}, policy:{tf.nn.softmax(output.policy)}')
					if len(self.observation_history)>=2:
						recurrent_output=out[1]
						print(f'recurrent: value:{recurrent_output.value},reward:{recurrent_output.reward}/{reward}, policy:{tf.nn.softmax(recurrent_output.policy)}\n')


				last_index=index+1
				assert line[last_index]=='[',f'get illegal first word:in:{line}get:\'{line[last_index]}\''
				index=line.find(']',last_index)+1
				self.child_visits.append(eval(line[last_index:index]))
				
				last_index=index+1
				self.root_values.append(eval(line[last_index:]))
		self.length=len(self.root_values)
	def add(self, action, observation, reward):
		self.action_history.append(action)
		self.observation_history.append(observation)
		self.reward_history.append(reward)
		self.length+=1
	def addtile(self, action):
		self.initial_add.append(action)
	def __str__(self):
		return f'observation:\n{self.observation_history}\n\naction:\n{self.action_history}\n\nreward:\n{self.reward_history}\n\nchild_visits:\n{self.child_visits}\n\nvalue:\n{self.root_values}'
class MCTS:
	"""
	Core Monte Carlo Tree Search algorithm.
	To decide on an action, we run N simulations, always starting at the root of
	the search tree and traversing the tree according to the UCB formula until we
	reach a leaf node.
	"""

	def __init__(self, config:my_config.Config, predictor:network.Predictor):
		self.config = config
		self.predictor=predictor
		self.sem=asyncio.Semaphore(config.search_threads)
		self.now_expanding=set()
		self.expanded=set()
	def run(self,
			observation, 
			legal_actions,
			add_exploration_noise,
			override_root_with=None,
			debug=True):
		"""
		At the root of the search tree we use the representation function to obtain a
		hidden state given the current observation.
		We then run a Monte Carlo Tree Search using only action sequences and the model
		learned by the network.
		"""
		if override_root_with:
			root = override_root_with
			root_predicted_value = None
		else:
			#defaulted not to use previously searched nodes and create a new tree
			root = Node(0)
			self.predictor.manager.add_coroutine_list(self.predictor.initial_inference(observation))
			output=self.predictor.manager.run_coroutine_list(True)[0]
			root_predicted_value=output.value
			reward=output.reward
			policy_logits=output.policy
			hidden_state=output.hidden_state
			assert len(legal_actions)>0, 'Legal actions should not be an empty array.'
			#only for check if actions are legal(unnecessary)
			flag=1
			for action in legal_actions:
				if action<0 or action>=4:
					flag=0
					break
			assert flag, f'Legal actions should be a subset of the action space. Got {legal_actions}'
			policy=tf.nn.softmax(policy_logits)
			if debug:
				print(f'reward:{reward}\nroot_predicted_value:{root_predicted_value}\npolicy:{policy}')
			root.expand(
				legal_actions,
				reward,
				policy,
				hidden_state,
			)

		if add_exploration_noise:
			root.add_exploration_noise(
				dirichlet_alpha=self.config.root_dirichlet_alpha,
				exploration_fraction=self.config.root_exploration_fraction,
			)

		min_max_stats = MinMaxStats()

		#max_tree_depth = 0
		for _ in range(self.config.num_simulations):
			self.predictor.manager.add_coroutine_list(self.tree_search(root, min_max_stats))
		self.predictor.manager.run_coroutine_list()
		#extra_info = {
		#	'root_predicted_value': root_predicted_value,
		#}#sometimes useful for debugging or playing?
		return root
	async def tree_search(self, node, min_max_stats):#->int|None:
		async with self.sem:
			now_expanding = self.now_expanding

			search_path=[node]
			current_tree_depth = 0
			while node.expanded():
				current_tree_depth += 1
				action, node = self.select_child(node, min_max_stats)
				search_path.append(node)
				while node in now_expanding:
					await asyncio.sleep(1e-4)
			self.now_expanding.add(node)
			# Inside the search tree we use the dynamics function to obtain the next hidden
			# state given an action and the previous hidden state
			parent=search_path[-2]
			output=await self.predictor.recurrent_inference(
				parent.hidden_state,
				action
			)
			node.expand(
				self.config.action_space_type0,
				output.reward,
				tf.nn.softmax(output.policy),
				output.hidden_state,
			)
			self.backpropagate(search_path, output.value, min_max_stats)
			self.now_expanding.remove(node)
		

	def select_child(self, node, min_max_stats):
		"""
		Select the child with the highest UCB score for mcts, not for final play.
		So type 1 contains all possible positions.
		"""
		action=None
		ucb=[self.ucb_score(node, child, min_max_stats) for child in node.children.values()]
		max_ucb = max(ucb)
		action = np.random.choice(
			[
				action
				for i,action in enumerate(node.children.keys())
				if ucb[i]==max_ucb
			]
		)
		return action, node.children[action]

	def ucb_score(self, parent, child, min_max_stats):#only for type0
		"""
		The score for a node is based on its value, plus an exploration bonus based on the prior.
		"""
		pb_c = (
			math.log(
				(parent.visit_count + self.config.pb_c_base + 1) / self.config.pb_c_base
			)
			+ self.config.pb_c_init
		)
		pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)

		prior_score = pb_c * child.prior

		if child.visit_count > 0:
			# Mean value Q
			value_score = min_max_stats.normalize(
				child.reward
				+ self.config.discount
				* child.value()
			)
		else:
			value_score = 0

		return prior_score + value_score

	def backpropagate(self, search_path, value, min_max_stats):
		"""
		At the end of a simulation, we propagate the evaluation all the way up the tree
		to the root.
		"""
		for node in reversed(search_path):
			node.value_sum += value
			node.visit_count += 1
			min_max_stats.update(node.reward + self.config.discount * node.value())

			value = node.reward + self.config.discount * value

		  
class Node:
	def __init__(self, prior):
		self.visit_count = 0
		self.prior = prior
		self.value_sum = 0
		self.children = {}
		self.hidden_state = None
		self.reward = 0

	def expanded(self):
		return len(self.children) > 0

	def value(self):
		if self.visit_count == 0:
			return 0
		return self.value_sum / self.visit_count

	def expand(self, actions, reward, policy_logits, hidden_state):
		"""
		We expand a node using the value, reward and policy prediction obtained from the
		neural network.
		"""
		assert type(actions) in (int,list), f'type(actions)=type({actions})={type(actions)}, not int or list'
		if type(actions)==int:
			actions=list(range(actions))
		self.reward = reward
		self.hidden_state = hidden_state
		policy_values = tf.nn.softmax(
			[policy_logits[a] for a in actions]
		).numpy()
		policy = {a: policy_values[i] for i, a in enumerate(actions)}
		for action, p in policy.items():
			self.children[action] = Node(p)

	def add_exploration_noise(self, dirichlet_alpha, exploration_fraction):
		"""
		At the start of each search, we add dirichlet noise to the prior of the root to
		encourage the search to explore new actions.
		"""
		actions = list(self.children.keys())
		
		noise = np.random.dirichlet([dirichlet_alpha] * len(actions))
		frac = exploration_fraction
		for a, n in zip(actions, noise):
			self.children[a].prior = self.children[a].prior * (1 - frac) + n * frac
class SelfPlay:
	"""
	Class which run in a dedicated thread to play games and save them to the replay-buffer.
	"""

	def __init__(self, predictor:network.Predictor, Game:type, config:my_config.Config):
		#have to pass seed because each self_play_worker should get different random seed to play different games
		self.config = config
		self.debug = config.debug
		self.add_exploration_noise=config.if_add_exploration_noise
		self.Game = Game

		# Initialize the network
		# should initialize manager, predictor at main.py, all selfplayer(self_play_worker*num_actors and test_worker*1)
		self.predictor=predictor

	def self_play(self, replay_buffer, shared_storage, test_mode=False):
		if test_mode:
			# Take the best action (no exploration) in test mode
			# This is for log(to tensorboard), in order to see the progress
			game_history = self.play_game(
				0,
				True,
			)

			# Save to the shared storage
			shared_storage.set_info(
				{
					"game_length": len(game_history.action_history) - 1,#first history is initial state
					"total_reward": sum(game_history.reward_history),#final score in case of 2048
					"stdev_reward": np.std(game_history.reward_history),
				}
			)
			return
		total=0
		while total<self.config.selfplay_games_per_run:
			#self.predictor.manager.set_weights(shared_storage.get_info("weights"))

			print('flag self_play1')
			game_history=self.play_game(
				self.config.visit_softmax_temperature_fn(
					training_steps=shared_storage.get_info("training_step")
				),
				True,### if you want to render, change this
			)
			print('flag self_play2')
			replay_buffer.save_game(game_history)#error seems to be here
			print('flag self_play3')
			total+=1
	
	def play_game(self, temperature, render:bool, game_id:int=-5):#for this single game, seed should be self.seed+game_id
		"""
		Play one game with actions based on the Monte Carlo tree search at each moves.
		"""
		game_history = GameHistory()
		# start a whole new game
		game=self.Game(self.config)
		game.reset()
		observation = game.get_features(None)
		#initial position
		#training target can be started at a time where the next move is adding move, so keep all observation history

		for _ in range(2):
			action=game.add()
			observation=game.get_features(1)
			game_history.addtile(action)
		done = False

		if render:
			print('A new game just started.')
			game.render()
		now_type=0
		while not done and len(game_history.action_history) <= self.config.max_moves:
			print('flag play_game1')
			if now_type==0:
				assert (
					len(np.array(observation).shape) == 3
				), f"Observation should be 3 dimensionnal instead of {len(np.array(observation).shape)} dimensionnal. Got observation of shape: {np.array(observation).shape}"
				assert (
					list(observation.shape) == self.config.observation_shape
				), f"Observation should match the observation_shape defined in MuZeroConfig. Expected {self.config.observation_shape} but got {np.array(observation).shape}."
				'''#This will only be useful if 
				stacked_observations = game_history.get_stacked_observations(
					-1,
					self.config.stacked_observations,
				)
				'''
				# Choose the action
				legal_actions=game.legal_actions()
				root = MCTS(self.config, self.predictor).run(
					observation,
					legal_actions,
					True,
				)
				action = self.select_action(
					root,
					temperature,
				)

				if render:
					#print(f'Tree depth: {mcts_info["max_tree_depth"]}')
					print(
						f"Root value : {root.value():.2f}"
					)
					print(f'visits:{[int(root.children[i].visit_count/self.config.num_simulations*100) if i in root.children else 0 for i in range(4)]}')
				reward = game.step(action)
				observation=game.get_features(0)
				if render:
					print(f"Played action: {environment.action_to_string(action,self.config.board_size)}")

				game_history.store_search_statistics(root, self.config.action_space_type0, self.config.winer_takes_all)


				#add a tile
				addaction=game.add()
				game_history.add([action,addaction],observation,reward)
				if render:
					game.render()
			
			done=game.finish()
			print(f'game length:{len(game_history.root_values)}')
			print('flag play_game2')
			now_type = 0 if now_type==1 else 1
		print('flag play_game3')
		return game_history

	def select_action(self, node, temperature):
		"""
		Select action according to the visit count distribution and the temperature.
		The temperature is changed dynamically with the visit_softmax_temperature function
		in the config.
		this function always choose from actions with type==0
		"""
		visit_counts = np.array(
			[child.visit_count for child in node.children.values()], dtype=np.float32
		)
		actions = [action for action in node.children.keys()]
		if temperature == 0:
			action = actions[np.argmax(visit_counts)]
		elif temperature == float("inf"):
			action = np.random.choice(actions)
		else:
			# See paper appendix Data Generation
			visit_count_distribution = visit_counts ** (1 / temperature)
			visit_count_distribution = visit_count_distribution / sum(
				visit_count_distribution
			)
			action = np.random.choice(actions, p=visit_count_distribution)

		return action
#show a game(for debugging)
if __name__=='__main__':
	con=my_config.default_config()
	net=network.Network(con)
	print(net.representation_model.summary())
	exit(0)
	weights=pickle.load(open('results/2022-04-02--14-58-12/model-001316.pkl', "rb"))
	net.set_weights(weights)
	manager=network.Manager(con,net)
	pre=network.Predictor(manager,con)
	his=GameHistory()
	his.load('saved_games/resnet/132.record',con,pre)
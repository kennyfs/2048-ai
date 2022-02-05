import asyncio
import collections
import math
import random
import sys
import time
from typing import Tuple

import numpy as np
import ray
import tensorflow as tf
import environment
import network
import config
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
class GameHistory:####
	"""
	Store only usefull information of a self-play game.
	at time t, gamehistory[t] should contain the information about:
	observation, value, child_visits at time t
	action, reward, action type from t-1 to t
	"""
	#states followed by actions of type x = "type x state"
	def __init__(self):
		self.observation_history = []
		self.action_history = []
		self.reward_history = []
		self.type_history = []
		self.child_visits = []
		self.root_values = []
		self.length = 0
		self.reanalysed_predicted_root_values = None
		# For PER(not in my plan)
		# self.priorities = None
		# self.game_priority = None

	def store_search_statistics(self, root, action_space):
		'''
		Turn visit count from root into a policy, store policy and valuesss
		'''
		if root is not None:
			sum_visits = sum(child.visit_count for child in root.children.values())
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
			self.root_values.append(None)

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
			for action,reward,type,visits,value in zip(self.action_history,self.reward_history,self.type_history,self.child_visits,self.root_values):
				F.write(f'{action} {reward} {type} {visits} {value}\n')
	def add(self, action, observation, reward, _type):
		self.action_history.append(action)
		self.observation_history.append(observation)
		self.reward_history.append(reward)
		self.type_history.append(_type)
		self.length+=1
class MCTS:
	"""
	Core Monte Carlo Tree Search algorithm.
	To decide on an action, we run N simulations, always starting at the root of
	the search tree and traversing the tree according to the UCB formula until we
	reach a leaf node.
	"""

	def __init__(self, config, predictor):
		self.config = config
		self.push_queue = predictor.manager.push_queue_func
		self.manager = predictor.manager
		self.sem=asyncio.Semaphore(config.search_threads)
		self.now_expanding=set()
		self.expanded=set()
	async def run(self,
			predictor,
			observation, 
			legal_actions,
			now_type,
			add_exploration_noise,
			override_root_with=None):
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
			output=await predictor.initial_inference(observation)
			
			root_predicted_value=output.value
			reward=output.reward
			policy_logits=output.policy
			hidden_state=output.hidden_state
			assert len(legal_actions)>0, 'Legal actions should not be an empty array.'
			flag=1
			for action in legal_actions:
				if action<0 or action>=4:
					flag=0
					break
			assert flag, f'Legal actions should be a subset of the action space. Got {legal_actions}'
			root.expand(
				legal_actions,
				now_type,
				reward,
				policy_logits,
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
			self.manager.add_coroutine_list(self.tree_search(root, now_type, min_max_stats))
		depths=self.manager.run_coroutine_list()
		max_tree_depth=max(depths)
		extra_info = {
			'max_tree_depth': max_tree_depth,
			'root_predicted_value': root_predicted_value,
		}#sometimes useful for debugging or playing?
		return root, extra_info
	async def tree_search(self, node, now_type, min_max_stats)->int:
		###Independent MCTS, run one simulation###
		self.running_simulation_num += 1

		# reduce parallel search number
		with await self.sem:
			depth = await self.start_tree_search(node, now_type, min_max_stats)
			self.running_simulation_num -= 1

		return depth
	async def start_tree_search(self, node, now_type, min_max_stats)->int:
		now_expanding = self.now_expanding

		while node in now_expanding:
			await asyncio.sleep(1e-4)
		search_path=[node]
		current_tree_depth = 0
		#now_type is always the type of action to this node
		while node.expanded():
			current_tree_depth += 1
			action, node = self.select_child(node, now_type, min_max_stats)
			search_path.append(node)
			now_type=1 if now_type==0 else 0
			
		# Inside the search tree we use the dynamics function to obtain the next hidden
		# state given an action and the previous hidden state
		parent=search_path[-2]
		output=await self.predictor.recurrent_inference(
			parent.hidden_state,
			np.array([action])
		)
		if now_type==1:
			actions=self.config.action_space_type1_size
			output.policy=self.config.type1_p
		else:
			actions=self.config.action_space_type0_size
		node.expand(
			actions,
			now_type,
			output.reward,
			output.policy,
			output.hidden_state,
		)

		self.backpropagate(search_path, output.value, now_type, min_max_stats)

		return current_tree_depth

		

	def select_child(self, node, now_type, min_max_stats):
		"""
		Select the child with the highest UCB score for mcts, not for final play.
		So type 1 contains all possible positions.
		"""
		action=None
		if now_type==1:#this can be really randomly choosing one, or based on ucb.
			#randomly
			p=[child.prior for child in node.children.values()]
			assert sum(p)==1
			action=np.random.choice(list(node.children.keys()),p=p)
		else:
			ucb=[self.ucb_score(node, child, min_max_stats) for action,child in node.children.items()]
			max_ucb = max(ucb)
			action = np.random.choice(
				[
					action
					for i,action in enumerate(node.children.keys())
					if ucb[i]==max_ucb
				]
			)
		assert action!=None,f'action not decided in MCTS.select_child!, with now_type={now_type}'
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
				* (child.value() if len(self.config.players) == 1 else -child.value())
			)
		else:
			value_score = 0

		return prior_score + value_score

	def backpropagate(self, search_path, value, to_play, min_max_stats):
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

	def expand(self, actions, now_type, reward, policy_logits, hidden_state):
		"""
		We expand a node using the value, reward and policy prediction obtained from the
		neural network.
		"""
		assert type(actions) in (int,list), f'type(actions)=type({actions})={type(actions)}, not int or list'
		if type(actions)==int:
			actions=list(range(actions))
		self.reward = reward
		self.hidden_state = hidden_state
		#for type 1(adding a tile), policy:{2:9,4:1}, it will choose randomly
		if now_type==1:
			for action in actions:
				self.children[action] = Node(policy_logits[action])
			return
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

	def __init__(self, initial_checkpoint, Game, config:config.Config):
		self.config = config
		self.debug = config.debug
		seed=config.seed
		self.add_exploration_noise=config.if_add_exploration_noise
		if seed==None:
			seed=random.randrange(sys.maxsize)
			if self.debug:
				print(f'seed was set to be {seed}.')
		self.game = environment.Environment(seed)####### temporarily set for vs code check

		# Fix random generator seed
		np.random.seed(seed)
		tf.random.set_seed(seed)

		# Initialize the network
		self.model = network.Network(config)# -> AbstractNetwork
		#### should initialize manager, predictor here
		manager=network.Manager(config, self.model)
		manager.set_weights(initial_checkpoint["weights"])
		self.predictor=network.Predictor(manager)
	def continuous_self_play(self, shared_storage, replay_buffer, test_mode=False):
		while (ray.get(shared_storage.get_info.remote("training_step")) < self.config.training_steps
				)and(
				not ray.get(shared_storage.get_info.remote("terminate"))):
			self.model.set_weights(ray.get(shared_storage.get_info.remote("weights")))

			if test_mode:
				raise NotImplementedError				
				pass ####
				# Take the best action (no exploration) in test mode
				'''
				game_history = self.play_game(
					0,
					self.config.temperature_threshold,
					False,
				)

				# Save to the shared storage
				shared_storage.set_info.remote(
					{
						"episode_length": len(game_history.action_history) - 1,
						"total_reward": sum(game_history.reward_history),
						"mean_value": np.mean(
							[value for value in game_history.root_values if value]
						),
					}
				)
				if 1 < len(self.config.players):
					shared_storage.set_info.remote(
						{
							"muzero_reward": sum(
								reward
								for i, reward in enumerate(game_history.reward_history)
								if game_history.to_play_history[i - 1]
								== self.config.muzero_player
							),
							"opponent_reward": sum(
								reward
								for i, reward in enumerate(game_history.reward_history)
								if game_history.to_play_history[i - 1]
								!= self.config.muzero_player
							),
						}
					)
				'''
			else:
				game_history = self.play_game(
					self.config.visit_softmax_temperature_fn(
						trained_steps=shared_storage.get_info
					),
					False,### if want to render, change this
				)

				replay_buffer.save_game.remote(game_history, shared_storage)
				

			# Managing the self-play / training ratio
			if not test_mode and self.config.self_play_delay:
				time.sleep(self.config.self_play_delay)
			if not test_mode and self.config.ratio:
				while (
					shared_storage.get_info.remote("training_step") / max(
						1, shared_storage.get_info.remote("num_played_steps")
					) < self.config.ratio####
					and 
					shared_storage.get_info.remote("training_step")
					< self.config.training_steps
					and 
					not shared_storage.get_info.remote("terminate")
				):
					time.sleep(0.5)

		self.close_game()

	def play_game(self, temperature, render):
		"""
		Play one game with actions based on the Monte Carlo tree search at each moves.
		"""
		game_history = GameHistory()
		# start a whole new game
		self.game.reset()
		#self.game should keep now_type
		observation = self.game.get_features()
		#initial position #### I'm not sure whether or not I should keep this
		game_history.add(None,observation,None,None)
		#training target can be started at a time where the next move is adding move, so keep all observation history

		for _ in range(2):
			action=self.game.add()
			observation=self.game.get_features()
			game_history.add(action,observation,None,1)
		for _ in range(3):
			game_history.root_values.append(None)
			game_history.child_visits.append(None)
		done = False

		if render:
			print('A new game just started.')
			self.game.render()
		assert self.game.now_type==0
		while not done and len(game_history.action_history) <= self.config.max_moves:
			if self.game.now_type==0:
				assert (
					len(np.array(observation).shape) == 3
				), f"Observation should be 3 dimensionnal instead of {len(np.array(observation).shape)} dimensionnal. Got observation of shape: {np.array(observation).shape}"
				assert (
					np.array(observation).shape == self.config.observation_shape
				), f"Observation should match the observation_shape defined in MuZeroConfig. Expected {self.config.observation_shape} but got {np.array(observation).shape}."
				'''#This will only be useful if 
				stacked_observations = game_history.get_stacked_observations(
					-1,
					self.config.stacked_observations,
				)
				'''
				# Choose the action
				legal_actions=self.game.legal_actions()
				root, mcts_info = MCTS(self.config).run(self.predictor,observation,legal_actions,self.game.now_type)
				action = self.select_action(
					root,
					temperature
				)

				if render:
					print(f'Tree depth: {mcts_info["max_tree_depth"]}')
					print(
						f"Root value : {root.value():.2f}"
					)
				reward = self.game.step(action)#type changed here
				done=self.game.finish()
				observation=self.game.get_features()
				if render:
					print(f"Played action: {self.game.action_to_string(action)}")
					self.game.render()

				game_history.store_search_statistics(root, self.config.action_space_type0)

				# Next batch
				game_history.add(action,observation,reward,0)
			else:
				action=self.game.add()
				self.game.change_type()
				observation=self.game.get_features()
				game_history.add(action,observation,None,1)
		return game_history

	def close_game(self):
		self.game.close()

	@staticmethod
	def select_action(node, temperature):
		"""
		Select action according to the visit count distribution and the temperature.
		The temperature is changed dynamically with the visit_softmax_temperature function
		in the config.
		this function always choose from actions with type==0
		"""
		visit_counts = np.array(
			[child.visit_count for child in node.children.values()], dtype="int32"
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

import asyncio
import collections
from copy import copy
import math
import pickle
import random
import sys

import numpy as np
from dataio.trainingwrite import GameData
import tensorflow as tf

import game.environment as environment
import neuralnet.model as model
from neuralnet.nnio import NNOutput

MAXIMUM_FLOAT_VALUE = float('inf')
KnownBounds = collections.namedtuple('KnownBounds', ['min', 'max'])

class MinMaxStats():
	"""A class that holds the min-max values of the tree."""

	def __init__(self, knownBounds=None):
		self.maximum=knownBounds.max if knownBounds else -MAXIMUM_FLOAT_VALUE
		self.minimum=knownBounds.min if knownBounds else MAXIMUM_FLOAT_VALUE
		
	def update(self, value:float):
		self.maximum=max(self.maximum,value)
		self.minimum=min(self.minimum,value)

	def normalize(self,value:float)->float:
		if self.maximum > self.minimum:
			# We normalize only when we have set the maximum and minimum values.
			return (value-self.minimum)/(self.maximum-self.minimum)
		return value
		
class SearchParams:
	def __init__(self,
		numSimulations:int,
		pbCBase:float,
		pbCInit:float,
		addExplorationNoise:bool,
		dirichletAlpha:float,
		explorationFraction:float,
		discount:float,
		):
		self.numSimulations = numSimulations
		self.pbCBase = pbCBase
		self.pbCInit = pbCInit
		self.addExplorationNoise = addExplorationNoise
		self.dirichletAlpha = dirichletAlpha
		self.explorationFraction = explorationFraction
		self.discount = discount

class Search:
	"""
	Designed for one thread, multiple workers. Does not create a class for thread.
	"""

	def __init__(self, board:environment.Board, nnEval, useNHWC:bool, params:SearchParams):
		self.root = None
		self.board = board
		self.nnEval = nnEval
		self.useNHWC = useNHWC
		self.actionSpace = board.actionSpacePlay()
		self.minMaxStats = MinMaxStats()
		self.params = params

	async def runWholeSearch(self):
		if self.root == None:
			self.root = Node(0)
		await self.treeSearch(self.root)
		self.root.addExplorationNoise(self.params.dirichletAlpha, self.params.explorationFraction)
		for _ in range(self.params.numSimulations - self.root.visits):
			await self.treeSearch(self.root)

	async def treeSearch(self, node):

		searchPath=[node]
		while node.expanded():
			action, node = self.selectChild(node)
			self.board.step(action)
			searchPath.append(node)

			action = self.board.add()
			node = node.children[action]
			searchPath.append(node)
		assert node is not None
		output = NNOutput()
		await self.nnEval.evaluate(
			self.board,
			output,
			self.useNHWC,
		)
		node.expand(
			self.actionSpace,
			tf.nn.softmax(output.policy)
		)
		self.backpropagate(searchPath, output.value)
		
	def selectChild(self, node):
		action = None
		ucb = [self.ucbScore(node, child) for child in node.children.values()]
		maxUcb = -MAXIMUM_FLOAT_VALUE
		for i,action in enumerate(node.children.keys()):# don't care if some nodes have same ucb value
			if ucb[i] > maxUcb:
				maxUcb = ucb[i]
				action = i
		return action, node.children[action]

	def ucbScore(self, parent, child):
		"""
		The score for a node is based on its value, plus an exploration bonus based on the prior.
		"""
		pbC = (
			math.log(
				(parent.visits + self.params.pbCBase + 1) / self.params.pbCBase
			)
			+ self.params.pbCInit
		)
		pbC *= math.sqrt(parent.visits) / (child.visits + 1)

		priorScore = pbC * child.prior

		if child.visits > 0:
			valueScore = self.minMaxStats.normalize(
				child.reward
				+ self.params.discount
				* child.value()
			)
		else:
			valueScore = 0

		return priorScore + valueScore

	def backpropagate(self, searchPath, value):
		"""
		At the end of a simulation, we propagate the evaluation all the way up the tree
		to the root.
		"""
		for node in reversed(searchPath):
			node.valueSum += value
			node.visits += 1
			self.minMaxStats.update(node.reward + self.discount * node.value())

			value = node.reward + self.discount * value

		  
class Node:
	def __init__(self, prior):
		self.visits = 0
		self.prior = prior
		self.valueSum = 0
		self.children = {}
		self.reward = 0

	def expanded(self):
		return len(self.children) > 0

	def value(self):
		if self.visits == 0:
			return 0
		return self.valueSum / self.visits

	def expand(self, actions, reward, policyLogits):
		assert type(actions) in (int,list), f'type(actions)=type({actions})={type(actions)}, not int or list'
		if type(actions)==int:
			actions=list(range(actions))
		self.reward = reward
		policy_values = tf.nn.softmax(
			[policyLogits[a] for a in actions]
		).numpy()
		policy = {a: policy_values[i] for i, a in enumerate(actions)}
		for action, p in policy.items():
			self.children[action] = Node(p)

	def addExplorationNoise(self, dirichletAlpha, explorationFraction):
		"""
		At the start of each search, we add dirichlet noise to the prior of the root to
		encourage the search to explore new actions.
		"""
		actions = list(self.children.keys())
		
		noise = np.random.dirichlet([dirichletAlpha] * len(actions))
		frac = explorationFraction
		for a, n in zip(actions, noise):
			self.children[a].prior = self.children[a].prior * (1 - frac) + n * frac

class SelfPlay:
	"""
	Class which run in a dedicated thread to play games and save them to the replay-buffer.
	"""

	def __init__(self, nnEval, Game:type, cfg):
		self.cfg = cfg

		numSimulations = cfg.getInt('numSimulations')
		pbCBase = cfg.getFloat('pbCBase')
		pbCInit = cfg.getFloat('pbCInit')
		addExplorationNoise = cfg.getBool('addExplorationNoise')
		dirichletAlpha = cfg.getFloat('dirichletAlpha')
		explorationFraction = cfg.getFloat('explorationFraction')
		discount = cfg.getFloat('discount')
		self.searchParams = SearchParams(numSimulations, pbCBase, pbCInit, addExplorationNoise, dirichletAlpha, explorationFraction, discount)
		self.seed = cfg.getString('searchSeed')
		if self.seed == None:
			self.seed = random.randrange(sys.maxsize)
			#### to do: log seed
		self.numSelfplayGamePerIteration = cfg.getInt('numSelfplayGamePerIteration')
		self.useNHWC = cfg.getBool('useNHWC')
		self.Game = Game

		self.nnEval = nnEval

	async def selfPlay(self, replayBuffer, sharedStorage, gameIdStart:int, render:bool=False):
		totalGames = 0
		for _ in range(self.numSelfplayGamePerIteration):

			gameData=self.playGame(
				self.cfg.temperatureFunction(
					trainingSteps=sharedStorage.getInfo("training step")
				)
			)
			replayBuffer.saveGame(gameData)
	
	async def playGame(self, temperature:float, render:bool):#for this single game, seed should be self.seed+game_id

		gameData = GameData()
		game = self.Game()
		game.reset()
		#initial position
		#training target can be started at a time where the next move is adding move, so keep all observation history

		for _ in range(2):
			action = game.add()
		gameData.observationHistory.append(copy(game.grid))
		done = False

		if render:
			print('A new game just started.')
			#### todo: change to logging
			game.render()
		search = Search(game, self.nnEval, self.useNHWC, self.searchParams)

		while not done and len(gameData.actionHistory) <= self.cfg.getInt('maxMoves', 1e10):

			
			# Choose the action
			legalActions = game.legalActions()
			root = await search.runWholeSearch()
			action = self.selectAction(
				root,
				temperature,
			)

			if render:
				print(
					f"Root value : {root.value():.2f}"
				)
				print(f'visits:{[int(root.children[i].visits/self.config.num_simulations*100) if i in root.children else 0 for i in range(4)]}')
				#### todo: change to logging
			reward = game.step(action)
			if render:
				print(f"Played action: {environment.actionToString(action,self.config.board_size)}")

			gameData.store_search_statistics(root, self.config.action_space_type0)


			#add a tile
			addaction=game.add()
			gameData.add([action,addaction],observation,reward)
			if render:
				game.render()
			
			done=game.finish()
			print(f'game length:{len(gameData.root_values)}')
			print('flag playGame2')
		print('flag playGame3')
		return gameData

	def selectAction(self, node, temperature):
		"""
		Select action according to the visit count distribution and the temperature.
		The temperature is changed dynamically with the visit_softmax_temperature function
		in the config.
		this function always choose from actions with type==0
		"""
		visitss = np.array(
			[child.visits for child in node.children.values()], dtype=np.float32
		)
		actions = [action for action in node.children.keys()]
		if temperature == 0:
			action = actions[np.argmax(visitss)]
		elif temperature == float("inf"):
			action = np.random.choice(actions)
		else:
			# See paper appendix Data Generation
			visits_distribution = visitss ** (1 / temperature)
			visits_distribution = visits_distribution / sum(
				visits_distribution
			)
			action = np.random.choice(actions, p=visits_distribution)

		return action
	def play_random_games(self, replay_buffer, shared_storage, render:bool=False):
		"""
		Play a game with a random policy.
		"""
		for game_id in range(self.config.num_random_games_per_iteration):
			gameHistory = GameHistory()
			game = self.Game(self.config)
			game.reset()
			done = False
			for _ in range(2):
				action=game.add()
				gameHistory.addtile(action)
			while not done and len(gameHistory.action_history) <= self.config.max_moves:
				action = np.random.choice(game.legal_actions())
				observation=game.get_features()
				reward = game.step(action)
				addaction=game.add()
				gameHistory.add([action,addaction],observation,reward)
				gameHistory.store_search_statistics(None, self.config.action_space_type0)
				done = game.finish()
				if render:
					game.render()
			if game_id%50==0:
				print(f'ratio:{game_id/self.config.num_random_games_per_iteration}')
			replay_buffer.save_game(gameHistory)
#show a game(for debugging)
if __name__=='__main__':
	net=model.Network()
	print(net.representation_model.summary())
	exit(0)
	weights=pickle.load(open('results/2022-04-02--14-58-12/model-001316.pkl', "rb"))
	net.set_weights(weights)
	manager=model.Manager(con,net)
	pre=model.Predictor(manager,con)
	his=GameHistory()
	his.load('saved_games/resnet/132.record',con,pre)
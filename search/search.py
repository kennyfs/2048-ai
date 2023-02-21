import math
import numpy as np

import tensorflow as tf

from game import environment
from neuralnet.nnio import NNOutput
from search.searchhelpers import MinMaxStats, SearchParams

MAXIMUM_FLOAT_VALUE = float("inf")


class Search:
    """
    Designed for one thread, multiple workers. Does not create a class for thread.
    """

    def __init__(
        self, board: environment.Board, nnEval, useNHWC: bool, params: SearchParams
    ):
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
        self.root.addExplorationNoise(
            self.params.dirichletAlpha, self.params.explorationFraction
        )
        for _ in range(self.params.numSimulations - self.root.visits):
            await self.treeSearch(self.root)

    async def treeSearch(self, node):
        searchPath = [node]
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
        node.expand(self.actionSpace, tf.nn.softmax(output.policy))
        self.backpropagate(searchPath, output.value)

    def selectChild(self, node):
        action = None
        ucb = [self.ucbScore(node, child) for child in node.children.values()]
        maxUcb = -MAXIMUM_FLOAT_VALUE
        for i, action in enumerate(
            node.children.keys()
        ):  # don't care if some nodes have same ucb value
            if ucb[i] > maxUcb:
                maxUcb = ucb[i]
                action = i
        return action, node.children[action]

    def ucbScore(self, parent, child):
        """
        The score for a node is based on its value, plus an exploration bonus based on the prior.
        """
        pbC = (
            math.log((parent.visits + self.params.pbCBase + 1) / self.params.pbCBase)
            + self.params.pbCInit
        )
        pbC *= math.sqrt(parent.visits) / (child.visits + 1)

        priorScore = pbC * child.prior

        if child.visits > 0:
            valueScore = self.minMaxStats.normalize(
                child.reward + self.params.discount * child.value()
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
        assert type(actions) in (
            int,
            list,
        ), f"type(actions)=type({actions})={type(actions)}, not int or list"
        if type(actions) == int:
            actions = list(range(actions))
        self.reward = reward
        policy_values = tf.nn.softmax([policyLogits[a] for a in actions]).numpy()
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

import numpy as np
from neuralnet.nnio import NNInput


class FinishedGameData:
    """
    Store only useful information of a self-play game.

    IMPORTANT!!!
    at time t, GameData[t] should contain the information about:
    grid, value, and childVisits at time t(that is, the grid before the action)
    action and reward produced from t-1 to t
    """

    # states followed by actions of type x = "type x state"
    def __init__(self):
        # each element is a grid [xSize][ySize]
        # initial board is unnecessary because self.gridHistory[0] is the same.
        self.gridHistory = []
        self.childVisits = []
        self.rootValues = []

        # if the game starts from scratch, self.gridHistory[0] has the initial two tiles, so which tiles are added is not important.
        self.actionHistory = []
        self.rewardHistory = []

    def storeSearchStatistics(self, root=None, actionSpace=None):
        """
        Turn visit count from root into a policy, store policy
        """
        if root is not None:
            sum_visits = sum(child.visits for child in root.children.values())
            self.childVisits.append(
                [
                    root.children[a].visits / sum_visits if a in root.children else 0
                    for a in actionSpace
                ]
            )

            self.rootValues.append(root.value())
        else:
            self.childVisits.append(
                [1 / len(actionSpace) for i in range(len(actionSpace))]
            )
            self.rootValues.append(0)

    def addRow(self, action, grid, reward):
        self.actionHistory.append(action)
        self.gridHistory.append(grid)
        self.rewardHistory.append(reward)

    def __str__(self):
        return f"gridHistory:\n{self.gridHistory}\n\nactionHistory:\n{self.actionHistory}\n\nrewardHistory:\n{self.rewardHistory}\n\nchildVisits:\n{self.childVisits}\n\nrootValues:\n{self.rootValues}"


class TrainingWriter:
    def __init__(self):
        self.games = []

    def saveToFile(self, filename):
        with open(filename, "wb") as f:
            nninput = NNInput()
            inputArray = []
            valueTarget = []
            for game in self.games:
                for t in range(len(game.gridHistory)):
                    grid = game.gridHistory[t]
                    nninput.fillV1(self.cfg.useNHWC, grid=grid)
                    inputArray.append(np.copy(nninput.spatialData))
                    valueTarget.append(self.computeValueTarget(game, t))

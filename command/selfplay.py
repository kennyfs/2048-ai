import pickle
import random
import sys
from copy import copy

import numpy as np

from dataio.trainingwrite import FinishedGameData
from game import environment
from neuralnet import model
from search.search import Search
from search.searchhelpers import SearchParams


class SelfPlay:
    """
    Class which run in a dedicated thread to play games and save them to the replay-buffer.
    """

    def __init__(self, nnEval, Game: type, cfg):
        self.cfg = cfg

        numSimulations = cfg.getInt("numSimulations")
        pbCBase = cfg.getFloat("pbCBase")
        pbCInit = cfg.getFloat("pbCInit")
        addExplorationNoise = cfg.getBool("addExplorationNoise")
        dirichletAlpha = cfg.getFloat("dirichletAlpha")
        explorationFraction = cfg.getFloat("explorationFraction")
        discount = cfg.getFloat("discount")

        self.searchParams = SearchParams(
            numSimulations,
            pbCBase,
            pbCInit,
            addExplorationNoise,
            dirichletAlpha,
            explorationFraction,
            discount,
        )
        self.seed = cfg.getString("searchSeed")
        if self.seed == None:
            self.seed = random.randrange(sys.maxsize)
            #### to do: log seed
        self.numSelfplayGamePerIteration = cfg.getInt("numSelfplayGamePerIteration")
        self.useNHWC = cfg.getBool("useNHWC")
        self.Game = Game

        self.nnEval = nnEval

    async def selfPlay(
        self, replayBuffer, sharedStorage, gameIdStart: int, render: bool = False
    ):
        totalGames = 0
        for _ in range(self.numSelfplayGamePerIteration):
            gameData = self.playGame(
                self.cfg.temperatureFunction(
                    trainingSteps=sharedStorage.getInfo("training step")
                )
            )
            replayBuffer.saveGame(gameData)

    async def playGame(
        self, temperature: float, render: bool
    ):  # for this single game, seed should be self.seed+game_id
        gameData = FinishedGameData()
        game = environment.Board()
        game.reset()
        actionSpace = game.actionSpace()  # action space in 2048 never changes
        # initial position
        # training target can be started at a time where the next move is adding move, so keep all observation history

        for _ in range(2):
            action = game.add()
        gameData.gridHistory.append(copy(game.grid))
        done = False

        if render:
            print("A new game just started.")
            #### todo: change to logging
            game.render()
        search = Search(game, self.nnEval, self.useNHWC, self.searchParams)

        while not done and len(gameData.actionHistory) <= self.cfg.getInt(
            "maxMoves", 1e10
        ):
            grid = copy(game.grid)
            # Choose the action
            root = await search.runWholeSearch()
            action = self.selectAction(
                root,
                temperature,
            )

            if render:
                print(f"Root value : {root.value():.2f}")
                print(
                    f"visits:{[int(root.children[i].visits/self.config.num_simulations*100) if i in root.children else 0 for i in range(4)]}"
                )
                #### todo: change to logging
            reward = game.step(action)
            if render:
                print(
                    f"Played action: {environment.actionToString(action,self.config.board_size)}"
                )

            gameData.storeSearchStatistics(root, actionSpace)

            # add a tile
            addaction = game.add()
            gameData.addRow([action, addaction], grid, reward)
            if render:
                game.render()

            done = game.finish()
            print(f"game length:{len(gameData.root_values)}")
            print("flag playGame2")
        print("flag playGame3")  #### todo: change to logging
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
            visits_distribution = visits_distribution / sum(visits_distribution)
            action = np.random.choice(actions, p=visits_distribution)

        return action

    def play_random_games(self, replay_buffer, shared_storage, render: bool = False):
        """
        Play a game with a random policy.
        """
        for game_id in range(self.config.num_random_games_per_iteration):
            gameHistory = FinishedGameData()
            game = self.Game(self.config)
            game.reset()
            done = False
            for _ in range(2):
                action = game.add()
                gameHistory.addtile(action)
            while not done and len(gameHistory.action_history) <= self.config.max_moves:
                action = np.random.choice(game.legal_actions())
                observation = game.get_features()
                reward = game.step(action)
                addaction = game.add()
                gameHistory.add([action, addaction], observation, reward)
                gameHistory.store_search_statistics(
                    None, self.config.action_space_type0
                )
                done = game.finish()
                if render:
                    game.render()
            if game_id % 50 == 0:
                print(f"ratio:{game_id/self.config.num_random_games_per_iteration}")
            replay_buffer.save_game(gameHistory)


# show a game(for debugging)
if __name__ == "__main__":
    net = model.Network()
    print(net.representation_model.summary())
    exit(0)
    weights = pickle.load(open("results/2022-04-02--14-58-12/model-001316.pkl", "rb"))
    net.set_weights(weights)
    manager = model.Manager(con, net)
    pre = model.Predictor(manager, con)
    his = GameHistory()
    his.load("saved_games/resnet/132.record", con, pre)

import copy
import random
import sys


class OutputHelper:
    BACKGROUND = "\x1b[48;5;"
    WORD = "\x1b[38;5;"
    END = "m"
    RESET = "\x1b[0m"

    def colorize(self, text, wordColor=None, backgroundColor=None):
        result = ""
        if wordColor is not None:
            result += self.WORD + str(wordColor) + self.END
        if backgroundColor is not None:
            result += self.BACKGROUND + str(backgroundColor) + self.END
        result += text
        result += self.RESET
        return result

    def tileToString(self, tile):
        background = [
            None,
            "252",
            "222",
            "208",
            "202",
            "166",
            "196",
            "227",
            "190",
            "184",
            "184",
            "220",
            "14",
        ][tile]
        word = [
            None,
            "234",
            "234",
            None,
            None,
            None,
            None,
            "234",
            "234",
            None,
            None,
            None,
            None,
        ][tile]
        return self.colorize("%3d" % 2**tile, word, background)


# class Action:
# 0~3:up,down,left,right
# starting from 4:put a tile, [4,4+self.board_size**2) for putting a 2, [4+self.board_size**2,4+2*self.board_size**2) for putting a 4
# (treat putting a tile as an action)
class Board:
    wordToAction = {"U": 0, "D": 1, "L": 2, "R": 3, "1": 0, "2": 1, "3": 2, "4": 3}

    def __init__(self, seed, cfg, board=None, score=None):
        """
        seed must be set
        """
        self.cfg = cfg
        self.boardSize = cfg.getInt("boardSize", -1)
        if self.boardSize == -1:
            self.xSize = cfg.getInt("xSize")
            self.ySize = cfg.getInt("ySize")
        else:
            self.xSize = self.boardSize
            self.ySize = self.boardSize
        if board is not None:
            self.grid = copy.deepcopy(board)
        else:
            self.reset()
        if score is not None:
            self.score = score
        else:
            self.score = 0

        if seed == None:
            seed = random.randrange(sys.maxsize)
            #### todo: log seed
        self.rand = random.Random(seed)

    def reset(self):
        """
        clear but not adding 2 tiles, as self_play should know where the initial tiles are
        """
        self.grid = [[0] * self.ySize for _ in range(self.xSize)]
        self.score = 0

    def moveALine(self, line, size, reverse):
        if reverse:
            line = line[::-1]  # reverse
        # move over all empty blocks
        nonZeros = 0
        for index in range(size):
            if line[index] != 0:
                if index > nonZeros:
                    line[nonZeros] = line[index]
                    line[index] = 0
                nonZeros += 1
        # combine same, adjacent tiles
        for i in range(size - 1):
            if line[i] == line[i + 1] and line[i] > 0:
                line[i] += 1
                line[i + 1] = 0
                self.score += 2 ** line[i]
        # move over all empty blocks(only pos 0,1,2 can be blank)
        nonZeros = 0
        for index in range(size - 1):
            if line[index] != 0:
                if index > nonZeros:
                    line[nonZeros] = line[index]
                    line[index] = 0
                nonZeros += 1
        if reverse:
            line = line[::-1]  # reverse
        return line

    def moveALineMaybeFaster(self, axis, otherAxisIdx, reverse):
        """
        axis: which axis the tiles moves along, 0 for x, 1 for y
        reverse: whether the line is reversed
        that is, up=(0,False), down=(0,True), left=(1,False), right=(1,True)
        """
        size = self.xSize if axis == 0 else self.ySize

        def get(idx):
            if axis == 0:
                if reverse:
                    return self.grid[size - idx][otherAxisIdx]
                else:
                    return self.grid[idx][otherAxisIdx]
            else:
                if reverse:
                    return self.grid[otherAxisIdx][size - idx]
                else:
                    return self.grid[otherAxisIdx][idx]

        def move(idxA, idxB):  # move idxA to idxB
            if axis == 0:
                if reverse:
                    self.grid[size - idxB][otherAxisIdx] = self.grid[size - idxA][
                        otherAxisIdx
                    ]
                    self.grid[size - idxA][otherAxisIdx] = 0
                else:
                    self.grid[idxB][otherAxisIdx] = self.grid[idxA][otherAxisIdx]
                    self.grid[idxA][otherAxisIdx] = 0
            else:
                if reverse:
                    self.grid[otherAxisIdx][size - idxB] = self.grid[otherAxisIdx][
                        size - idxA
                    ]
                    self.grid[otherAxisIdx][size - idxA] = 0
                else:
                    self.grid[otherAxisIdx][idxB] = self.grid[otherAxisIdx][idxA]
                    self.grid[otherAxisIdx][idxA] = 0

        def add1(idx):  # add 1 to idx
            if axis == 0:
                if reverse:
                    self.grid[size - idx][otherAxisIdx] += 1
                else:
                    self.grid[idx][otherAxisIdx] += 1
            else:
                if reverse:
                    self.grid[otherAxisIdx][size - idx] += 1
                else:
                    self.grid[otherAxisIdx][idx] += 1

        def clear(idx):  # clear idx
            if axis == 0:
                if reverse:
                    self.grid[size - idx][otherAxisIdx] = 0
                else:
                    self.grid[idx][otherAxisIdx] = 0
            else:
                if reverse:
                    self.grid[otherAxisIdx][size - idx] = 0
                else:
                    self.grid[otherAxisIdx][idx] = 0

        # move over all empty blocks
        nonZeros = 0
        for index in range(size):
            if get(index) != 0:
                if index > nonZeros:
                    move(index, nonZeros)
                nonZeros += 1
        # combine same, adjacent tiles
        for i in range(size - 1):
            if get(i) == get(i + 1) and get(i) > 0:
                add1(i)
                clear(i + 1)
                self.score += 2 ** get(i)
        # move over all empty blocks(only pos 0,1,2 can be blank)
        nonZeros = 0
        for index in range(size - 1):
            if get(index) != 0:
                if index > nonZeros:
                    move(index, nonZeros)
                nonZeros += 1

    def step(self, action) -> int:  # Actions are in order, up, down, left, and right.
        """
        return value: instant reward
        """
        beforescore = self.score
        if action == 0:
            for i in range(self.ySize):
                res = self.moveALine(
                    [self.grid[j][i] for j in range(self.xSize)], self.xSize, False
                )
                for j in range(self.xSize):
                    self.grid[j][i] = res[j]
        elif action == 1:
            for i in range(self.ySize):
                res = self.moveALine(
                    [self.grid[j][i] for j in range(self.xSize)], self.xSize, True
                )
                for j in range(self.xSize):
                    self.grid[j][i] = res[j]
        elif action == 2:
            for i in range(self.xSize):
                res = self.moveALine(
                    [self.grid[i][j] for j in range(self.ySize)], self.ySize, False
                )
                for j in range(self.ySize):
                    self.grid[i][j] = res[j]
        elif action == 3:
            for i in range(self.xSize):
                res = self.moveALine(
                    [self.grid[i][j] for j in range(self.ySize)], self.ySize, False
                )
                for j in range(self.ySize):
                    self.grid[i][j] = res[j]
        else:
            x, y, num = self.addActionToPos(action)
            self.grid[x][y] = num
        return self.score - beforescore

    def stepMaybeFaster(self, action) -> int:
        """
        return value: instant reward
        """
        beforescore = self.score
        if action == 0:
            for i in range(self.ySize):
                self.moveALineMaybeFaster(0, i, False)
        elif action == 1:
            for i in range(self.ySize):
                self.moveALineMaybeFaster(0, i, True)
        elif action == 2:
            for i in range(self.xSize):
                self.moveALineMaybeFaster(1, i, False)
        elif action == 3:
            for i in range(self.xSize):
                self.moveALineMaybeFaster(1, i, True)
        else:
            x, y, num = self.addActionToPos(action)
            self.grid[x][y] = num
        return self.score - beforescore

    def render(self):
        print(self)

    def isTerminate(self):
        for x in range(self.xSize):
            for y in range(self.ySize):
                if self.grid[x][y] == 0:
                    return False
                if x + 1 < self.xSize and self.grid[x][y] == self.grid[x + 1][y]:
                    return False
                if y + 1 < self.ySize and self.grid[x][y] == self.grid[x][y + 1]:
                    return False
        return True

    def legalActions(self):  # List[Action]
        return [i for i in range(4) if self.valid(i)]

    def getBlanks(self):  # List[(x,y)] where self[x][y] == 0
        return [
            (x, y)
            for x in range(self.board_size)
            for y in range(self.board_size)
            if self.grid[x][y] == 0
        ]

    def add(self) -> int:  # Action
        blank = self.get_blanks()
        if len(blank) == 0:
            self.render()
            raise BaseException("no blank in grid")
        x, y = blank[self.rand.randint(0, len(blank) - 1)]
        if self.rand.randint(1, 10) == 1:
            num = 2
        else:
            num = 1
        self.grid[x][y] = num
        return self.addPosToAction(x, y, num, self.board_size)

    def valid(self, action) -> bool:
        if 0 <= action and action <= 3:  ###### this needs optimizing
            tmp = Board(None, self.cfg, board=copy.deepcopy(self.grid))
            a = copy.deepcopy(tmp.grid)
            tmp.step(action)
            for x in range(self.board_size):
                for y in range(self.board_size):
                    if a[x][y] != tmp.grid[x][y]:
                        return True
            return False
        if 4 <= action and action <= 4 + 2 * self.board_size**2:
            action = (action - 4) % (self.board_size**2)
            x = action // self.board_size
            y = action % self.board_size
            return self.grid[x][y] == 0
        raise BaseException(
            "action(" + str(action) + ") out of range (in Environment.valid)"
        )

    def askHumanForAction(self):
        action = input("Enter a action(UDLR or 1234):")
        action = self.wordToAction[action]
        while self.valid(action) == False:
            action = input("Invalid action, try again:")
            action = self.wordToAction[action]
        return action

    def actionSpace(self):
        return list(range(4))

    def legalActions(self):
        return [i for i in range(4) if self.valid(i)]

    def addActionToPos(self, action: int):
        """
        return value:
                x, y, num(1 or 2)
        """
        assert action >= 4 and action < 4 + 2 * self.xSize * self.ySize
        action -= 4
        num = action // (self.xSize * self.ySize) + 1
        if action > self.xSize * self.ySize:
            action -= self.xSize * self.ySize
        return action // self.ySize, action % self.ySize, num

    def addPosToAction(self, x, y, num):
        return 4 + x * self.ySize + y + (num - 1) * self.xSize * self.ySize

    def actionToString(self, action: int) -> str:
        if self.actionType(action) == 0:
            return ["Up", "Down", "Left", "Right"][action]
        # type 1(add tile)
        x, y, num = self.addActionToPos(action)
        return f"Adding a {2**num} at {x},{y}"

    def actionType(self, action):
        assert (
            action >= 0 and action < 4 + 2 * self.xSize * self.ySize
        ), f"action is {action}"
        if action < 4:
            return 0
        return 1

    def __str__(self):
        result = ""
        for i in self.grid:
            for j in i:
                if j > 0:
                    result += OutputHelper.tileToString(j)
                else:
                    result += " " * 3
                result += "|"
            result += "\n"
            result += "-" * (4 * self.ySize)
            result += "\n"
        result += f"score = {self.score}"
        return result


def test():
    from program import setup

    cfg = setup.ConfigParser("/home/kenny/Desktop/python3/2048-ai/config/dev.cfg")
    board = Board(None, cfg)
    print(board)

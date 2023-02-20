from game import environment
import numpy as np

class NNInput:
	def __init__(self):
		self.spatialData = None
		# Unused, maybe useful in the future.
		self.globalData = None
		# None means that a symmetry will be chosen randomly by "NNEval",
		# so when running multiple search threads or workers,
		# the symmetries may not be the same due to the different order in which the inputs are received.
	def fillV1(self, useNHWC:bool, board:environment.Environment=None, grid=None):
		'''
		V1 only contains the current tiles, no history and no type info.
		Adding history might be an improvement.
		Add type then I can make a malicious tile adder.
		
		2048's input is terribly dependent on the board size. The number of channels it needs is in O(N^2), where N is the board size.
		So the whole input is in O(N^4). Nevertheless, this may not be a problem, because nobody would play a 2048 with a board size like 100.

		This class equals NNResultBuf in KataGo.

		if not useNHWC, use NCHW
		'''
		if board is not None:
			grid = board.grid
		else:
			assert grid is not None
		grid = np.array(grid)
		xSize, ySize = grid.shape
		result = []
		for i in range(1, xSize * ySize + 2):# 2^(x*y+1) is the max possible tile.
			result.append(np.where(grid == i, 1.0, 0.0))
		out = np.array(result, dtype = np.float32)
		if useNHWC:# CHW to HWC
			out = np.transpose(out, (1, 2, 0))
		self.spatialData = out
class NNOutput:
	pass
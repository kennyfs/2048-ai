class GameData:
	"""
	Store only useful information of a self-play game.
	
	IMPORTANT!!!
	at time t, GameData[t] should contain the information about:
	board, value, and childVisits at time t
	action and reward produced from t-1 to t
	"""
	#states followed by actions of type x = "type x state"
	def __init__(self):
        # each element is a grid [xSize][ySize]
        # initial board is unnecessary because self.observationHistory[0] is the same.
		self.observationHistory = []
		self.childVisits = []
		self.rootValues = []

        # if the game starts from scratch, self.observationHistory[0] has the initial two tiles, so which tiles are added is not important.
		self.actionHistory = []
		self.rewardHistory = []

	def storeSearchStatistics(self, root=None, actionSpace=None):
		'''
		Turn visit count from root into a policy, store policy
		'''
		if root is not None:
			sum_visits = sum(child.visits for child in root.children.values())
			self.childVisits.append(
				[
					root.children[a].visits / sum_visits
					if a in root.children
					else 0
					for a in actionSpace
				]
			)

			self.rootValues.append(root.value())
		else:
			self.childVisits.append([1/len(actionSpace) for i in range(len(actionSpace))])
			self.rootValues.append(0)

	def getObservation(self, index):
		index = index % len(self.observationHistory)
		return self.observationHistory[index]

	def addRow(self, action, board, reward):
		self.actionHistory.append(action)
		self.observationHistory.append(board)
		self.rewardHistory.append(reward)

	def __str__(self):
		return f'observationHistory:\n{self.observationHistory}\n\nactionHistory:\n{self.actionHistory}\n\nrewardHistory:\n{self.rewardHistory}\n\nchildVisits:\n{self.childVisits}\n\nrootValues:\n{self.rootValues}'
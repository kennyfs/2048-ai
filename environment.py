import numpy as np
from random import randint
from copy import deepcopy
bg   ="\x1b[48;5;"
word ="\x1b[38;5;"
end  ="m"
reset="\x1b[0m"

def get_features(grid)->np.array:#given grid, return features for Network input
	sz=len(grid)
	grid=np.array(grid)
	result=np.array([])
	for i in range(1,sz*sz+1):
		result=np.concatenate((result,np.where(grid==i,1.0,0.0)))
	return result
	
class Action:
	#0~3:up,down,left,right
	#starting from 4:put a tile, [4,4+self.board_size**2) for putting a 2, [4+self.board_size**2,4+2*self.board_size**2) for putting a 4
	#(treat putting a tile as an action)
	def __init__(self,index:int):
		self.index = index
	def __hash__(self):
		return self.index
	def __eq__(self, other):
		return self.index == other.index
	def __gt__(self, other):
		return self.index > other.index
class Environment:
	def __init__(self,config,board=None,score=None,g=None):
		self.g=g
		self.config=config
		self.board_size=config.board_size
		if board:#if it's not None
			self.grid=deepcopy(board)
		else:
			self.init()
		if score:
			self.score=score
		else:
			self.score=0
	def clear(self):
		self.grid=[[0]*self.board_size for _ in range(self.board_size)]
	def init(self):
		self.clear()
		self.score=0
	def movealine(self,line,reverse):#move from self.board_size-1 to 0
		if reverse:
			line=line[::-1]#reverse
		#move over all blank
		index=1
		while index<self.board_size:
			forward=index-1
			while forward>=0 and line[forward]==0:
				line[forward]=line[forward+1]
				line[forward+1]=0
				forward-=1
			index+=1
		#combine
		for i in range(self.board_size-1):
			if line[i]==line[i+1] and line[i]>0:
				line[i]+=1
				line[i+1]=0
				self.score+=2**line[i]
		#move over blank(only pos 1,2 can be blank)
		for i in range(1,self.board_size-1):
			if line[i]==0:
				line[i],line[i+1]=line[i+1],0
		if reverse:
			line=line[::-1]#reverse
		return line
	def step(self,action:Action):#up down left right
		beforescore=self.score
		if action.index==0:
			for i in range(self.board_size):
				res=self.movealine([self.grid[j][i] for j in range(self.board_size)],False)
				for j in range(self.board_size):
					self.grid[j][i]=res[j]
		elif action.index==1:
			for i in range(self.board_size):
				res=self.movealine([self.grid[j][i] for j in range(self.board_size)],True)
				for j in range(self.board_size):
					self.grid[j][i]=res[j]
		elif action.index==2:
			for i in range(self.board_size):
				res=self.movealine([self.grid[i][j] for j in range(self.board_size)],False)
				for j in range(self.board_size):
					self.grid[i][j]=res[j]
		elif action.index==3:
			for i in range(self.board_size):
				res=self.movealine([self.grid[i][j] for j in range(self.board_size)],True)
				for j in range(self.board_size):
					self.grid[i][j]=res[j]
		tmp=action.index
		if 0<=tmp and tmp<=3:
			return self.score-beforescore
			
		tmp-=4
		if 0<=tmp and tmp<self.board_size**2:
			self.grid[tmp//self.board_size][tmp%self.board_size]=1
			return 0
			
		tmp-=self.board_size**2
		if 0<=tmp and tmp<=self.board_size**2:
			self.grid[tmp//self.board_size][tmp%self.board_size]=2
			return 0
		raise BaseException('action('+str(action.index)+') out of range (in Environment.step)')
	def dump(self):
		for i in self.grid:
			for j in i:
				if j>0:
					print(bg+{1:'252'+end+word+'234',2:'222'+end+word+'234',3:'208',4:'202',5:'166',6:'196',7:'227',8:'190',9:'184',10:'184',11:'220',12:'14',}[j]+end+'%3d'%2**j+reset,end='')
				else:
					print('   ',end='')
				print('|',end='')
			print('\n','-'*4*self.board_size,sep='')
		print('score=',self.score)
	def finish(self):
		for x in range(self.board_size):
			for y in range(self.board_size):
				if self.grid[x][y]==0:
					return False
				for dx,dy in ((1,0),(0,1),(-1,0),(0,-1)):
					tx,ty=x+dx,y+dy
					if tx>self.board_size-1 or ty>self.board_size-1 or tx<0 or ty<0:
						continue
					if self.grid[x][y]==self.grid[tx][ty]:
						return False
		return True
	def legal_actions(self):#List[Action]
		result=[]
		for i in range(4):
			if self.valid(Action(i)):
				result.append(Action(i))
		return result
	def get_blanks(self):#List[(x,y)] where self[x][y]==0
		result=[]
		for x in range(self.board_size):
			for y in range(self.board_size):
				if self.grid[x][y]==0:
					result.append((x,y))
		return result
	def add(self)->Action:
		blank=self.get_blanks()
		if len(blank)==0:
			raise BaseException('no blank in grid (in Environment.add)')
		x,y=blank[randint(0,len(blank)-1)]
		v=2 if randint(1,10)==1 else 1#10% to be a 4(2)
		self.grid[x][y]=v
		return Action(4+(v-1)*self.board_size**2+x*self.board_size+y)
	def valid(self,action:Action)->bool:
		if 0<=action.index and action.index<=3:
			tmp=Environment(self.config,board=deepcopy(self.grid))
			a=deepcopy(tmp.grid)
			tmp.step(action)
			for x in range(self.board_size):
				for y in range(self.board_size):
					if a[x][y]!=tmp.grid[x][y]:
						return True
			return False
		if 4<=action.index and action.index<=4+2*self.board_size**2:
			action=(action-4)%(self.board_size**2)
			x=action//self.board_size
			y=action%self.board_size
			return self.grid[x][y]==0
		raise BaseException('action('+str(action.index)+') out of range (in Environment.valid)')

class Game:
	# In any game, the first two actions are adding tile, then each move is followed by adding tile
	def __init__(self,config):
		self.environment=Environment(config)  # Game specific environment.
		self.history=[]#List[Action]
		self.type=[]#0:move,1:adding tile # will also stored in Node
		self.child_visits=[]
		self.root_values=[]
		self.action_space_type0_size=config.action_space_type0_size
		self.action_space_type1_size=config.action_space_type1_size
		self.discount=config.discount
		
	def terminal(self)->bool:
		# if the game ends
		return self.environment.finish()
		
	def legal_actions(self)->'List[Action]':
		# list of legal actions, only care about move
		return self.environment.legal_actions()
		
	def apply(self,action):
		reward = self.environment.step(action)
		self.rewards.append(reward)
		self.history.append(action)
		self.type.append(1 if 0<=action.type and action.type<=3 else 0)
		
	def store_search_statistics(self, root):
		#only store type 0(move)
		#root.children is Dict[Node] whose keys are actions
		sum_visits = sum(child.visit_count for child in root.children.values())
		action_space = (Action(index) for index in range(self.action_space_type0_size))
		self.child_visits.append([
				root.children[a].visit_count / sum_visits if a in root.children else 0
				for a in action_space
		])
		self.root_values.append(root.value())
		#adding tile
		self.child_visits.append(None)
		self.root_values.append(None)
		
	def make_image(self, state_index: int):#-1 means the last
		# Game specific feature planes.
		# Go through history
		tmpenv=Environment()
		if state_index==-1:
			for action in self.history:
				tmpenv.step(action)
		elif 0<=state_index and state_index<=len(self.history):
			# do history[:state_index]
			for action in self.history[:state_index]:
				tmpenv.step(action)
		else:
			raise BaseException('state_index('+str(state_index)+') is out of range('+str(len(self.history))+') (in Game.make_image)')
		return get_features(tmpenv.grid)

	def make_target(self, state_index: int, num_unroll_steps: int, td_steps: int):#return type List[Tuple(value_target,reward_target,policy_target) policy might not have a sum of 1
		# The value target is the discounted root value of the search tree N steps
		# into the future, plus the discounted sum of all rewards until then.
		targets = []
		for current_index in range(state_index, state_index + num_unroll_steps + 1):
			bootstrap_index = current_index + td_steps
			if bootstrap_index>len(self.root_values)-1:
				break
			if self.type[bootstrap_index]==1:
				continue#only makes target for type0(move)
			value = self.root_values[bootstrap_index] * self.discount**td_steps
			dis=1
			for reward in self.rewards[current_index:bootstrap_index]:
				value += reward * dis	# pytype: disable=unsupported-operands
				dis*=self.discount

			# For simplicity the network always predicts the most recently received
			# reward, even for the initial representation network where we already
			# know this reward.
			if current_index > 0 and current_index <= len(self.rewards):
				last_reward = self.rewards[current_index - 1]
			else:
				last_reward = 0

			if current_index < len(self.root_values):
				targets.append((value, last_reward, self.child_visits[current_index]))
			else:
				# States past the end of games are treated as absorbing states.
				targets.append((0, last_reward, []))
		return targets
		
	def action_history(self):
		return ActionHistory(self.history, self.action_space_type0_size)
		
	def replay(self):
		tmp=Environment([[0]*4 for i in range(4)],addwhenplay=False)
		for i in self.history:
			sleep(0.8)
			tmp.step(i)
			tmp.dump()
			
	def write(self,f):
		with open(f,'w') as F:
			F.write(str(self.data))
			F.write('\n')
			F.write(str(self.moves))
			
	def read(self,f):#clear origin data, moves
		with open(f,'r') as F:
			self.data=eval(F.readline())
			self.moves=eval(F.readline())
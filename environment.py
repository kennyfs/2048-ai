from random import randint
from copy import deepcopy
bg   ="\x1b[48;5;"
word ="\x1b[38;5;"
end  ="m"
reset="\x1b[0m"
class Action:
	#0~3:up,down,left,right
	#starting from 4:put a tile, 4~19 for putting a 2, 20~35 for putting a 4
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
	def __init__(self,board=None,score=None,g=None):
		self.g=g
		if board:#if it's not None
			self.grid=deepcopy(board)
		else:
			self.init()
		if score:
			self.score=score
		else:
			self.score=0
	def clear(self):
		self.grid=[[0]*4 for _ in range(4)]
	def init(self):
		self.clear()
		self.score=0
	def movealine(self,line,reverse):#move from 3 to 0
		if reverse:
			line=line[::-1]#reverse
		#move over all blank
		index=1
		while index<4:
			forward=index-1
			while forward>=0 and line[forward]==0:
				line[forward]=line[forward+1]
				line[forward+1]=0
				forward-=1
			index+=1
		#combine
		for i in range(3):
			if line[i]==line[i+1] and line[i]>0:
				line[i]+=1
				line[i+1]=0
				self.score+=2**line[i]
		#move over blank(only pos 1,2 can be blank)
		for i in range(1,2+1):
			if line[i]==0:
				line[i],line[i+1]=line[i+1],0
		if reverse:
			line=line[::-1]#reverse
		return line
	def step(self,action:Action):#up down left right
		beforescore=self.score
		if action.index==0:
			for i in range(4):
				self.grid[0][i],self.grid[1][i],self.grid[2][i],self.grid[3][i]=self.movealine([self.grid[0][i],self.grid[1][i],self.grid[2][i],self.grid[3][i]],False)
		elif action.index==1:
			for i in range(4):
				self.grid[0][i],self.grid[1][i],self.grid[2][i],self.grid[3][i]=self.movealine([self.grid[0][i],self.grid[1][i],self.grid[2][i],self.grid[3][i]],True)
		elif action.index==2:
			for i in range(4):
				self.grid[i][0],self.grid[i][1],self.grid[i][2],self.grid[i][3]=self.movealine([self.grid[i][0],self.grid[i][1],self.grid[i][2],self.grid[i][3]],False)
		elif action.index==3:
			for i in range(4):
				self.grid[i][0],self.grid[i][1],self.grid[i][2],self.grid[i][3]=self.movealine([self.grid[i][0],self.grid[i][1],self.grid[i][2],self.grid[i][3]],True)
		if 0<=action.index and action.index<=3:
			return self.score-beforescore
		if 4<=action.index and action.index<=19:
			action.index-=4
			self.grid[action.index//4][action.index%4]=1
		if 20<=action.index and action.index<=35:
			action.index-=20
			self.grid[action.index//4][action.index%4]=2
		if action.index<0 or action.index>35:
			raise BaseException('action('+str(action.index)+') out of range [0,35] (in Environment.step)')
		return 0
	def dump(self):
		for i in self.grid:
			for j in i:
				if j>0:
					print(bg+{1:'252'+end+word+'234',2:'222'+end+word+'234',3:'208',4:'202',5:'166',6:'196',7:'227',8:'190',9:'184',10:'184',11:'220',12:'14',}[j]+end+'%3d'%2**j+reset,end='')
				else:
					print('   ',end='')
				print('|',end='')
			print('\n----------------')
		print('score=',self.score)
	def finish(self):
		for x in range(4):
			for y in range(4):
				if self.grid[x][y]==0:
					return False
				for dx,dy in ((1,0),(0,1),(-1,0),(0,-1)):
					tx,ty=x+dx,y+dy
					if tx>3 or ty>3 or tx<0 or ty<0:
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
		for x in range(4):
			for y in range(4):
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
		return Action(x*4+y+4+16*(v-1))
	def valid(self,action:Action)->bool:
		if 0<=action.index and action.index<=3:
			tmp=Environment(deepcopy(self.grid))
			a=deepcopy(tmp.grid)
			tmp.step(action.index)
			for x in range(4):
				for y in range(4):
					if a[x][y]!=tmp.grid[x][y]:
						return True
			return False
		if 4<=action.index and action.index<=35:
			action=(action-4)%16
			x=action//4
			y=action%4
			return self.grid[x][y]==0
		raise BaseException('action('+str(action.index)+') out of range [0,35] (in Environment.valid)')

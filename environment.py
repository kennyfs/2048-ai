from random import randint
from copy import deepcopy
bg   ="\x1b[48;5;"
word ="\x1b[38;5;"
end  ="m"
reset="\x1b[0m"
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
		self.add()
		self.add()
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
	def step(self,action):#up down left right
		beforescore=self.score
		if action==0:
			for i in range(4):
				self.grid[0][i],self.grid[1][i],self.grid[2][i],self.grid[3][i]=self.movealine([self.grid[0][i],self.grid[1][i],self.grid[2][i],self.grid[3][i]],False)
		elif action==1:
			for i in range(4):
				self.grid[0][i],self.grid[1][i],self.grid[2][i],self.grid[3][i]=self.movealine([self.grid[0][i],self.grid[1][i],self.grid[2][i],self.grid[3][i]],True)
		elif action==2:
			for i in range(4):
				self.grid[i][0],self.grid[i][1],self.grid[i][2],self.grid[i][3]=self.movealine([self.grid[i][0],self.grid[i][1],self.grid[i][2],self.grid[i][3]],False)
		elif action==3:
			for i in range(4):
				self.grid[i][0],self.grid[i][1],self.grid[i][2],self.grid[i][3]=self.movealine([self.grid[i][0],self.grid[i][1],self.grid[i][2],self.grid[i][3]],True)
		if 0<=action and action<=3:
			return self.score-beforescore
		if 4<=action and action<=19:
			action-=4
			self.grid[action//4][action%4]=1
		if 20<=action and action<=35:
			action-=20
			self.grid[action//4][action%4]=2
		if action<0 or action>35:
			raise BaseException('action('+str(action)+') out of range [0,35]')
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
	def legal_actions(self):#List[(x,y)]
		result=[]
		for x in range(4):
			for y in range(4):
				if self.grid[x][y]==0:
					result.append((x,y))
		return result
	def add(self):
		blank=self.legal_actions()
		if len(blank)==0:
			raise BaseException('no blank in grid (doing Environment.add)')
		i=randint(0,len(blank)-1)
		x,y=blank[i]
		v=randint(1,10)
		if v==1:#10%
			v=2
		else:
			v=1
		self.grid[x][y]=v
		return x,y,v
	def valid(self,action):
		tmp=Environment(deepcopy(self.grid))
		a=deepcopy(tmp.grid)
		tmp.step(action)
		for x in range(4):
			for y in range(4):
				if a[x][y]!=tmp.grid[x][y]:
					return True
		return False

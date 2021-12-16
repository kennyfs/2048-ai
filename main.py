from random import randint,random
from copy import copy,deepcopy
#from network import nn
import numpy as np
from time import time,sleep
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
	def step(self,drc):#up down left right
		if drc==0:
			for i in range(4):
				self.grid[0][i],self.grid[1][i],self.grid[2][i],self.grid[3][i]=self.movealine([self.grid[0][i],self.grid[1][i],self.grid[2][i],self.grid[3][i]],False)
		elif drc==1:
			for i in range(4):
				self.grid[0][i],self.grid[1][i],self.grid[2][i],self.grid[3][i]=self.movealine([self.grid[0][i],self.grid[1][i],self.grid[2][i],self.grid[3][i]],True)
		elif drc==2:
			for i in range(4):
				self.grid[i][0],self.grid[i][1],self.grid[i][2],self.grid[i][3]=self.movealine([self.grid[i][0],self.grid[i][1],self.grid[i][2],self.grid[i][3]],False)
		elif drc==3:
			for i in range(4):
				self.grid[i][0],self.grid[i][1],self.grid[i][2],self.grid[i][3]=self.movealine([self.grid[i][0],self.grid[i][1],self.grid[i][2],self.grid[i][3]],True)
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
	def getblank(self)->List[(x,y)]:
		result=[]
		for x in range(4):
			for y in range(4):
				if self.grid[x][y]==0:
					result.append((x,y))
		return result
	def add(self):
		blank=self.getblank()
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
		tmp=board(deepcopy(self.grid))
		a=deepcopy(tmp.grid)
		tmp.moveall(action)
		for x in range(4):
			for y in range(4):
				if a[x][y]!=tmp.grid[x][y]:
					return True
		return False
class Game:
	def __init__(self,action_space_size:int,discount:float,player='Human',modelinfo=None):
		self.info={'player':player,'modelinfo':modelinfo}
		self.environment=Environment()  # Game specific environment.
		self.history=[]
		self.child_visits=[]
		self.root_values=[]
		self.action_space_size=action_space_size
		self.discount=discount
	def addrand(self,x,y,v):
		self.moves.append((2,x,y,v))
	def step(self,action):
		self.moves.append((1,action))
	def replay(self):
		tmp=board([[0]*4 for i in range(4)],addwhenplay=False)
		for i in self.moves:
			sleep(0.5)
			if i[0]==1:
				tmp.step(i[1])
			else:
				tmp.grid[i[1]][i[2]]=i[3]
			tmp.dump()
			sleep(0.5)
	def write(self,f):
		with open(f,'w') as F:
			F.write(str(self.data))
			F.write('\n')
			F.write(str(self.moves))
	def read(self,f):#clear origin data, moves
		with open(f,'r') as F:
			self.data=eval(F.readline())
			self.moves=eval(F.readline())
def player():
	g=game()
	b=board(g=g)
	while not b.finish():
		b.dump()
		done=False
		while not done:
			valid=False
			while not valid:
				valid=True
				try:a=int(input())
				except:valid=False
				if a>3 or a<0:
					valid=False
			if b.valid(a):
				b.step(a)
				x,y,v=b.add()
				g.addrand(x,y,v)
				done=True
		
	b.dump()
	g.write('test.sgf')
#replay
g=Game()
g.read('test.sgf')
g.replay()
#play
player()

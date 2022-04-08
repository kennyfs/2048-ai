import random
import sys
import copy
from random import randint
from time import sleep

import numpy as np

bg   ="\x1b[48;5;"
word ="\x1b[38;5;"
end  ="m"
reset="\x1b[0m"
	
#class Action:
	#0~3:up,down,left,right
	#starting from 4:put a tile, [4,4+self.board_size**2) for putting a 2, [4+self.board_size**2,4+2*self.board_size**2) for putting a 4
	#(treat putting a tile as an action)
class Environment:
	def __init__(self,config,board=None,score=None):
		self.config=config
		self.board_size=config.board_size
		if board:#if it's not None
			self.grid=copy.deepcopy(board)
		else:
			self.reset()
		if score:
			self.score=score
		else:
			self.score=0
		#only changed in self.step, directly doing self.add should not change it.
		self.debug=config.debug
		seed=config.seed
		if seed==None:
			seed=random.randrange(sys.maxsize)
			if self.debug:
				print(f'seed was set to be {seed}.')
	def reset(self):
		'''
		clear but not adding 2 tiles, as self_play should know where the initial tiles are
		'''
		self.grid=[[0]*self.board_size for _ in range(self.board_size)]
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
	def step(self,action)->int:#up down left right
		'''
		input: action
		output: instant reward
		'''
		beforescore=self.score
		if action==0:
			for i in range(self.board_size):
				res=self.movealine([self.grid[j][i] for j in range(self.board_size)],False)
				for j in range(self.board_size):
					self.grid[j][i]=res[j]
		elif action==1:
			for i in range(self.board_size):
				res=self.movealine([self.grid[j][i] for j in range(self.board_size)],True)
				for j in range(self.board_size):
					self.grid[j][i]=res[j]
		elif action==2:
			for i in range(self.board_size):
				res=self.movealine([self.grid[i][j] for j in range(self.board_size)],False)
				for j in range(self.board_size):
					self.grid[i][j]=res[j]
		elif action==3:
			for i in range(self.board_size):
				res=self.movealine([self.grid[i][j] for j in range(self.board_size)],True)
				for j in range(self.board_size):
					self.grid[i][j]=res[j]
		
		return self.score-beforescore
	def render(self):
		for i in self.grid:
			for j in i:
				if j>0:
					print(bg+{1:'252'+end+word+'234',2:'222'+end+word+'234',3:'208',4:'202',5:'166',6:'196',7:'227'+end+word+'234',8:'190'+end+word+'234',9:'184',10:'184',11:'220',12:'14',}[j]+end+'%3d'%2**j+reset,end='')
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
				for dx,dy in ((1,0),(0,1)):
					tx,ty=x+dx,y+dy
					if tx>self.board_size-1 or ty>self.board_size-1:
						continue
					if self.grid[x][y]==self.grid[tx][ty]:
						return False
		return True
	def legal_actions(self):#List[Action]
		return [i for i in range(4) if self.valid(i)]
	def get_blanks(self):#List[(x,y)] where self[x][y]==0
		return [(x,y) for x in range(self.board_size) for y in range(self.board_size) if self.grid[x][y]==0]
	def add(self)->int:#Action
		blank=self.get_blanks()
		if len(blank)==0:
			self.render()
			raise BaseException('no blank in grid')
		x,y=blank[randint(0,len(blank)-1)]
		num=2 if randint(1,10)==1 else 1#10% to be a 4(2)
		self.grid[x][y]=num
		return add_pos_to_action(x,y,num,self.board_size)
	def valid(self,action)->bool:
		if 0<=action and action<=3:#### this needs optimizing
			tmp=Environment(self.config,board=copy.deepcopy(self.grid))
			a=copy.deepcopy(tmp.grid)
			tmp.step(action)
			for x in range(self.board_size):
				for y in range(self.board_size):
					if a[x][y]!=tmp.grid[x][y]:
						return True
			return False
		if 4<=action and action<=4+2*self.board_size**2:
			action=(action-4)%(self.board_size**2)
			x=action//self.board_size
			y=action%self.board_size
			return self.grid[x][y]==0
		raise BaseException('action('+str(action)+') out of range (in Environment.valid)')
	def get_features(self,now_type=None)->np.array:#given grid, return features for Network input
		#10^4 times per second
		if now_type==None:
			now_type=0
		grid=np.array(self.grid)
		result=[]
		for i in range(1,self.board_size**2+1):#board_size**2 is max possible tile
			result.append(np.where(grid==i,1.0,0.0))
		return np.array(result,dtype=np.float32)
		'''
		#alternative:
		6*10^3 times per second
		result=np.concatenate([np.expand_dims(np.where(grid==i,1.0,0.0),0) for i in range(1,self.board_size**2+1)],axis=0)
		return result
		'''

def add_pos_to_action(x,y,num,board_size):
	return 4+x*board_size+y+(num-1)*board_size**2
def action_to_string(action:int,board_size)->str:
	if action_to_type(action,board_size)==0:
		return (['Up','Down','Left','Right'])[action]
	#type 1(add tile)
	x,y,num=add_action_to_pos(action,board_size)
	return f'Adding a {2**num} at {x},{y}'
def action_to_type(action,board_size):
	assert action>=0 and action<4+2*board_size**2, f'action is {action}'# very likely impossible
	if action<4:
		return 0
	return 1
def add_action_to_pos(Action:int,board_size):
	'''
	return type:
		x,y,num(1 or 2)
	'''
	assert 4<=Action and Action<4+2*board_size**2
	Action-=4
	num=Action//(board_size**2)
	Action%=board_size**2
	return Action//board_size,Action%board_size,num+1
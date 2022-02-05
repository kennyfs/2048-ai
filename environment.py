import random
import sys
from copy import deepcopy
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
	def __init__(self,config,board=None,now_type=0,score=None):
		self.config=config
		self.board_size=config.board_size
		if board:#if it's not None
			self.grid=deepcopy(board)
		else:
			self.reset()
		if score:
			self.score=score
		else:
			self.score=0
		self.now_type=now_type#default is type 0, move
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
		self.now_type=0
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
		assert (0<=action<4 and self.now_type==0) or (action>=4 and self.now_type==1), f'get action {action}, but now_type is {self.now_type}.'
		self.change_type()
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
		if self.action_to_type(action)==0:
			return self.score-beforescore
		x,y,num=self.add_action_to_pos(action)
		self.grid[x][y]=num
		return 0
	def render(self):
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
		return self.add_pos_to_action(x,y,num)
	def valid(self,action)->bool:
		if 0<=action and action<=3:#### this needs optimizing
			tmp=Environment(self.config,board=deepcopy(self.grid))
			a=deepcopy(tmp.grid)
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
	def get_features(self)->np.array:#given grid, return features for Network input
		#10^4 times per second
		grid=np.array(self.grid)
		result=[]
		for i in range(1,self.board_size**2+1):#board_size**2 is max possible tile
			result.append(np.where(grid==i,1.0,0.0))
		return np.array(result)
		'''
		#alternative:
		6*10^3 times per second
		result=np.concatenate([np.expand_dims(np.where(grid==i,1.0,0.0),0) for i in range(1,self.board_size**2+1)],axis=0)
		return result
		'''
		
	def change_type(self):
		self.now_type=1 if self.now_type==0 else 1
	def add_action_to_pos(self,Action:int):
		'''
		return type:
			x,y,num(1 or 2)
		'''
		assert 4<=Action and Action<4+2*self.board_size**2
		Action-=4
		num=Action//(self.board_size**2)
		Action%=self.board_size**2
		return Action//self.board_size,Action%self.board_size,num+1
	def add_pos_to_action(self,x,y,num):
		return 4+x*self.board_size+y+(num-1)*self.board_size**2
	def action_to_string(self,action:int)->str:
		if self.action_to_type(action)==0:
			return (['Up','Down','Left','Right'])[action]
		#type 1(add tile)
		x,y,num=self.add_action_to_pos(action)
		return f'Adding a {2**num} at {x},{y}'
	def action_to_type(self,action):
		assert action>=0 and action<4+2*self.board_size**2, f'action is {action}'# very likely impossible
		if action<4:
			return 0
		return 1
if __name__=='__main__':
	import config,time
	my_config=config.default_config()
	g=Environment(my_config,[[1,2,3,4],[0,0,0,0],[1,2,6,2],[1,2,5,1]])
	start=time.time()
	for _ in range(100000):
		a=g.get_features()
	print(a)
	print(time.time()-start)
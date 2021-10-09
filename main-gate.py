from random import randint,random
from copy import copy,deepcopy
from network import nn
import numpy as np
from time import time
from itertools import chain
bg   ="\x1b[48;5;"
word ="\x1b[38;5;"
end  ="m"
reset="\x1b[0m"
class board:
	def __init__(self,board=None):
		if board:#if it's not None
			self.grid=board
		else:
			self.grid=[[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]
			self.add()
			self.add()
		self.score=0
		self.last_move=None
	def init(self):
		self.grid=[[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]
		self.score=0
	def movealine(self,line,reverse):
		if reverse:
			line=[line[3],line[2],line[1],line[0]]#reverse
		#move
		index=1
		while index<4:
			forward=index-1
			while forward>=0 and line[forward]==0:
				line[forward]=line[forward+1]
				line[forward+1]=0
				forward-=1
			index+=1
		#combine
		if line[0]==line[1] and line[0]>0:
			line[0]+=1
			line[1]=0
			self.score+=2**line[0]
		if line[1]==line[2] and line[1]>0:
			line[1]+=1
			line[2]=0
			self.score+=2**line[1]
		if line[2]==line[3] and line[2]>0:
			line[2]+=1
			line[3]=0
			self.score+=2**line[2]
		#move
		for i in range(1,3):
			if line[i]==0:
				line[i],line[i+1]=line[i+1],0
		if reverse:
			line=[line[3],line[2],line[1],line[0]]#reverse
		return line
	def moveall(self,drc):#up down left right
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
		if self.last_move!=None:
			print({0:'up',1:'down',2:'left',3:'right'}[self.last_move])
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
				for i in ((1,0),(0,1),(-1,0),(0,-1)):
					tx,ty=x+i[0],y+i[1]
					if tx>3 or ty>3 or tx<0 or ty<0:
						continue
					if self.grid[x][y]==0 or self.grid[x][y]==self.grid[tx][ty]:
						return False
		return True
	def add(self):
		none=[]
		for x in range(4):
			for y in range(4):
				if self.grid[x][y]==0:
					none.append((x,y))
		if none==[]:
			return False
		i=randint(0,len(none)-1)
		v=randint(1,10)
		if v==1:#10%
			v=2
		else:
			v=1
		self.grid[none[i][0]][none[i][1]]=v
		return True
	def play(self,direction):
		self.last_move=direction
		self.moveall(direction)
		return self.add()
	def ok(self,direction):
		tmp=board(deepcopy(self.grid))
		a=deepcopy(tmp.grid)
		tmp.moveall(direction)
		if a==tmp.grid:
			return False
		return True
def selfplay(nn,game_num,batch):#because I should predict many games in one go to save time, it'll be a little complicated.
	predata=[]
	movesdata=[]
	scores={}
	boards=[]
	newindex=0
	running=0
	finish_num=0
	while finish_num<game_num:
		while running<batch and finish_num+running<game_num:
		#	print(boards)
			predata.append([])
			movesdata.append([])
			b=board()
			boards.append([newindex,b])
			newindex+=1
			running+=1
		net_inputs=[]
		for gameid,aboard in boards:#so I have to remove the board after a game is finished.
			net_inputs.append([])#batch(axis0)
			for i in aboard.grid:
				net_inputs[-1].append([])#boardx(axis1)
				for j in i:
					net_inputs[-1][-1].append([])#boardy(axis2)
					for num in range(18):
						net_inputs[-1][-1][-1].append(int(j==num))
			predata[gameid].append(net_inputs[-1])
		net_inputs=np.array(net_inputs).astype('float32')
		moves=nn.predict(net_inputs)
		#print(moves)
		random=np.random.rand(running)
		play=[]
		#decide what to play
		for a,num in zip(random,range(running)):#alist is something like [0.2,0.4,0.3,0.3], which the sum is always 1.0
			'''
			alist=moves[num]
			num=alist[0]
			index=0
			for i in range(1,4):
				if alist[i]>num:
					index=i
			play.append(index)
			'''
			alist=moves[num]
			num=0
			for i in range(4):
				num+=alist[i]
				if a<=num:
					p=i
					break
			play.append(p)
			#	if boards[num].ok(p):
			#		play.append(p)
			#		break
			#	alist[num]=0.
			#	total=sum(alist)
			#	for i in range(4):
			#		alist[i]/=total
			#this sampling is not needed, the Neural network should learn from it(data will be based on "score/moves")
		#play
		end_games=[]
		index=0
		for i,loop_board in boards:
			movesdata[i].append(play[index])
			if not loop_board.play(play[index]):#it can't add any tile after move#punish the ai because it doesn't choose the better move(which can consume tiles and continue to play)
				scores[i]=loop_board.score
				finish_num+=1
				end_games.append(i)
				running-=1
			index+=1
		#if end_games!=[]:print(boards)
		boards=[[x,y] for x,y in boards if x not in end_games]
		#if end_games!=[]:print(boards)
		#for i in range(len(boards)):
		#	if boards[i][0] in end_games:
		#		del boards[i]
			#Then I will add a new game at the start of loop
		#judge whether the game is ended
		end_games=[]
		for i,loop_board in boards:
			if loop_board.finish():
				scores[i]=loop_board.score
				finish_num+=1
				end_games.append(i)
				running-=1
		#if end_games!=[]:print(boards)
		boards=[[x,y] for x,y in boards if x not in end_games]
		#if end_games!=[]:print(boards)
		#for i in end_games:
		#	del boards[i]
	#	if len(boards)>0 and boards[0][0]==0:
	#		boards[0][1].dump()
	#	time.sleep(0.2)
	'''
	data=[]
	for i in predata:
		data.extend(i)
	data=np.array(data)
	print(data.shape)
	'''
	scores=np.array([v for k, v in sorted(scores.items(), key=lambda item: item[0])])
	#data's shape is (games,moves of that game)
	#movesdata's shape is (games,moves([0..3]))
	#scores' shape is (games)
	return predata,movesdata,scores
nn=nn(init=False)
nn.load('nosearch-gate0.h5')
def test():
	_,_2,scores=selfplay(nn,1000,1000)
	scores=[v for k, v in sorted(scores.items(), key=lambda item: item[0])]
	scores=np.array(scores)
	print(_,'mean:',np.mean(scores),'stddev:',np.std(scores),end='')
def get(game_num,batch):
	data,moves,scores=selfplay(nn,game_num,batch)
	std=np.std(scores)
	print('mean',np.mean(scores),'std',std)
	normalscores=(scores-np.mean(scores))/std
	i=0
	todel=[]
	while i<len(normalscores):
		if normalscores[i]<0:
			todel.append(i)
		i+=1
	for index in sorted(todel, reverse=True):
		del moves[index]
		del data[index]
#	print((game_num-len(todel))/float(game_num)*100,'% of data remains.')
	scores=np.delete(scores,todel,0)
#	print(len(data),len(moves),len(scores))
	return data,moves,scores
def train(game_num,batch,times):
	for _ in range(times):
		data=[]
		moves=[]
		scores=[]
		sum_games=0
		while sum_games<game_num:
			start=time()
			datal,movesl,scoresl=get(game_num,game_num)
			sum_games+=len(scoresl)
		#	print('No.',_,game_num/(time()-start),'games/sec,mean:',np.mean(scoresl),'stddev:',np.std(scoresl),'\nsumgames=',sum_games,'the passing games',len(scoresl))
			print('sum_games',sum_games)
			data.extend(datal)
			moves.extend(movesl)
			scores=np.concatenate((scores,scoresl),axis=0)
		label=[]
		std=np.std(scores)
		for i in range(len(scores)):
			for j in range(len(moves[i])):
				label.append([])
				for k in range(4):
					if k==moves[i][j]:
						label[-1].append(1.)
					else:
						label[-1].append(0)
		label=np.asarray(label,dtype=np.float32)
		print('label num:',label.shape[0])
		finaldata=[]
		for i in data:
			finaldata.extend(i)
		data=np.asarray(finaldata,dtype=np.float32)
		nn.train(data,label,int(std/40))
		nn.save('nosearch-gate0.h5')
train(1000,1000,10)

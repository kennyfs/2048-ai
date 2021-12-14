from random import randint,random
from copy import copy,deepcopy
#from network import nn
import numpy as np
from time import time,sleep
from itertools import chain
TRAIN=0
bg   ="\x1b[48;5;"
word ="\x1b[38;5;"
end  ="m"
reset="\x1b[0m"
class game:
	def __init__(self,player='Human',modelinfo=None):
		self.data={'player':player,'modelinfo':modelinfo}
		self.moves=[]
	def addrand(self,x,y,v):
		self.moves.append((2,x,y,v))
	def play(self,direction):
		self.moves.append((1,direction))
	def replay(self):
		tmp=board([[0]*4 for i in range(4)],addwhenplay=False)
		for i in self.moves:
			sleep(0.5)
			if i[0]==1:
				tmp.play(i[1])
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
class board:
	def __init__(self,board=None,score=None,g=None,addwhenplay=True):
		self.g=g
		if board:#if it's not None
			self.grid=board
		else:
			self.init()
		if score:
			self.score=score
		else:
			self.score=0
		self.last_move=None
		self.addwhenplay=addwhenplay
	def init(self):
		self.grid=[[0]*4 for i in range(4)]
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
		#move
		for i in range(1,2+1):
			if line[i]==0:
				line[i],line[i+1]=line[i+1],0
		if reverse:
			line=line[::-1]#reverse
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
				if self.grid[x][y]==0:
					return False
				for i in ((1,0),(0,1),(-1,0),(0,-1)):
					tx,ty=x+i[0],y+i[1]
					if tx>3 or ty>3 or tx<0 or ty<0:
						continue
					if self.grid[x][y]==self.grid[tx][ty]:
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
		if self.g:
			self.g.addrand(none[i][0],none[i][1],v)
		return True
	def play(self,direction):
		self.last_move=direction
		self.moveall(direction)
		if self.g:
			self.g.play(direction)
		if self.addwhenplay:
			self.add()
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
						net_inputs[-1][-1][-1].append(float(j==num))
			predata[gameid].append(net_inputs[-1])
		net_inputs=np.array(net_inputs).astype('float32')
		moves=nn.predict(net_inputs)
		#print(moves)
		random=np.random.rand(running)
		play=[]
		#decide what to play
		for num,a in enumerate(random):#alist is something like [0.2,0.4,0.3,0.3], which the sum is always 1.0
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
	data=[]
	for i in predata:
		data.extend(i)
	data=np.array(data)
	scores=np.array([v for k, v in sorted(scores.items(), key=lambda item: item[0])])
	#data's shape is (games,moves of that game)
	#movesdata's shape is (games,moves([0..3]))
	#scores' shape is (games)
	return data,movesdata,scores
if TRAIN:
	nn=nn(init=True)
#nn.load('nosearch.h5')
def test():
	_,_2,scores=selfplay(nn,1000,1000)
	scores=[v for k, v in sorted(scores.items(), key=lambda item: item[0])]
	scores=np.array(scores)
	print(_,'mean:',np.mean(scores),'stddev:',np.std(scores),end='')
def train(game_num,batch,times):
	for _ in range(times):
		start=time()
		data,moves,scores=selfplay(nn,game_num,batch)
		print('No.',_,game_num/(time()-start),'games/sec,mean:',np.mean(scores),'stddev:',np.std(scores),end='')
		std=np.std(scores)
		scores=(scores-np.mean(scores))/std
		goal=0.5/(1+np.exp(-scores))
		#print(scores,goal)
		label=[]
		for i in range(len(scores)):
			v_move=goal[i]
			v_other=(1-v_move)/3
			#print(v_move,v_other)
			for j in range(len(moves[i])):
				label.append([])
				for k in range(4):
					if k==moves[i][j]:
						label[-1].append(v_move)
					else:
						label[-1].append(v_other)
		label=np.array(label)
		print(label.shape[0])
		data=np.asarray(data,dtype=np.float32)
		nn.train(data,label,30)
		nn.save('nosearchnogate.h5')
if TRAIN:
	train(1000,1000,10)
def player():
	g=game()
	b=board(g=g)
	for _ in range(10):
		b.dump()
		done=False
		while not done:
			ok=False
			while not ok:
				ok=True
				try:a=int(input())
				except:ok=False
				if a>3 or a<0:
					ok=False
			if b.ok(a):
				b.play(a)
				done=True
	b.dump()
	g.write('test.sgf')
#replay
g=game()
g.read('test.sgf')
g.replay()
#play
player()

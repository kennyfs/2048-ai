from random import randint,random
from copy import copy,deepcopy
from network import nn
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
		self.score=0
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
		i=randint(0,3)
		j=randint(0,3)
		while self.grid[i][j]!=0:
			i=randint(0,3)
			j=randint(0,3)
		self.grid[i][j]=randint(1,2)
	def play(self,direction):
		self.moveall(direction)
		self.add()
	def ok(self,direction):
		tmp=board(deepcopy(self.grid))
		a=deepcopy(tmp.grid)
		tmp.moveall(direction)
		if a==tmp.grid:
			return False
		return True
def selfplay(nn,games_num,batch):#because I should predict many games in one go to save time, it's a little complicated.#unfinished, too complex, will fix
	predata=[]
	boards=[]
	status=[]
	dataindexes=[]
	newdataindex=0
	for i in range(batch):
		predata.append([])
		boards.append(board())
		status.append(True)#if TRUE, finished
		dataindexes.append([])
		newdataindex+=1
	finish_num=0
	running=0
	while finish_num<games_num:#a loop, play a move in all games
		while running<batch and finish_num+running<games_num:
			index=status.index(True)
			boards[index].init()
			predata.append([])
			status[index]=False
			dataindexes.append(newdataindex)
			running+=1
			newdataindex+=1
		net_inputs=[]
		for aboard in boards:#so I have to remove the board after a game is finished.
			net_inputs.append([])#batch(axis0)
			for i in aboard.grid:
				net_inputs[-1].append([])#boardx(axis1)
				for j in i:
					net_inputs[-1][-1].append([])#boardy(axis2)
					for num in range(18):
						net_inputs[-1][-1][-1].append(int(j==num))
		net_inputs=np.array(net_inputs).astype('float32')
		moves=nn.predict(net_inputs)
		play=[]
		for num in range(len(boards)):#alist is something like [0.2,0.4,0.3,0.3]
			alist=moves[num]
			p=None
			while True:
				a=random()
				if a<alist[0]:
					p=0
				else:
					a-=alist[0]
				if a<alist[1]:
					p=1
				else:
					a-=alist[1]
				if a<alist[2]:
					p=2
				else:
					p=3
				if boards[num].ok(p):
					play.append(p)
					break
				alist[num]=0.
				total=sum(alist)
				for i in range(4):
					alist[i]/=total
		for num in len(boards):#so I have to remove the board after a game is finished.
			predata[dataindexes[num]].append([net_inputs[num],play[num]])
			boards[num].play(play[num])
			if boards[num].finish():
				if finish_num+running<games_num:
					running-=1
					status[num]=True
				else:
					...

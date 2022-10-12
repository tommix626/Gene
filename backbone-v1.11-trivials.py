# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 23:05:24 2021
version v1.11
(2021年2月10日)-(2021年2月11日)

The true start of the project.
The backbone starts to set up all the structure without programming into details.

//log
    0.8     start of file
            the structure of GameAgent
            more completion on the structure map(in OneNote)
            communicate with zyc in txhy and take "style" into account(not implemented->v1.1)
    0.9     finish the basic forward propagation step
            finish AI & GameAgent structural codes
            able to play with basic rules
    0.91    add testing for a specific agent,showing tangible results
    1.0     add player vs AI system
    1.1     add AI database
            improve test function(multi game)
    1.11    improve family naming system
            add full-featured block: contest
            add function clear_agent_pts for evolution_init
    
    
    
//ideas
    -use numpy not pytorch
    -max rounds (finish)

//probs
    -go into deadcycles when running Gameagent.game function ,reason unclear(->solved)
    -very weirdly,the AI's have mostly reached exactly half winning and half loosing
@author: Tom
@coauthor: wyk,xzj,yhy,zyc,lzc
"""
#import sys,pygame
import numpy as np
#import traceback #for raising error

#global functions
def throw_error(x):
  raise Exception(x)#异常被抛出
def tanh(x):
    return np.tanh(x)
def softmax(x):
    exp=np.exp(x-x.max())
    return exp/exp.sum()

#global variables&consts
ENV_LENGTH=50
OPTION_NUM=6
dimensions=[ENV_LENGTH,OPTION_NUM]
distribution=[
    {'b':[0,1]},#distribution[0],第零层的参数范围 FIXME
    {'b':[0,1],'w':[-30,30]}#FIXME 
]
activation=[tanh,softmax]# activation function of each layer
AI_NUM=eval(input("the amount of AI playing:")) # the amount of AI playing
TEST_AI_NUM=20 # v0.91 another serveral agents for testing a player-choosed AI
ACTION=6  # the num of choice to act
judge_matrix=[[0,-1,0,0,-1,-1],[1,0,0,0,-1,-1],[0,0,0,0,-1,-1],[0,0,0,0,0,0],[1,1,1,0,0,-1],[1,1,1,0,1,0]]
                #given acts,decide the state of the first player
                #(judge_matrix[act1][act2]=1/0 means whether player 1 win this round)
energy_cost=[1,-1,0,-1,-2,-3]
actionDict={"e":0,"k1":1,"k2":4,"k3":5,"d1":2,"d2":3}   #for PvC use(player input) (v1.0)
actionDict_reverse={0:"Collect Energy",1:"Simple Shot",4:"Double Kill",5:"Triple Kill",2:"Plain Defense",3:"Shielding"} #same(player output)
WINNING_PTS=100 #FIXME
#classes of objects
class CLS_AI(object):
    def __init__(self,family,code=0):#v1.11 code：编号，随着变异增加
        self.family_withoutcode=family
        self.paraList=[]#dictionary with struct:[{'b':...},{'w':...,'b':...}]
        self.pts=0
        self.pos=-1#when playing the pos(0 or 1),which is used to indicate which parameter in the env is self-based
        self.database_ns = {0:[0,0,0,0,0,0],1:[0,0,0,0,0,0],2:[0,0,0,0,0,0],3:[0,0,0,0,0,0],4:[0,0,0,0,0,0],5:[0,0,0,0,0,0],-1:[0,0,0,0,0,0]}#下一步(next step)的总数, -1 to record the first round v1.1
        self.database_tot = [0,0,0,0,0,0]#各action总数 v1.1
        self.currentact=-1#v1.1 for easier way to collect data
        self.lastact=-1#v1.1 same
        self.code = code
        self.family=str(family)+str(code)
    def init_parameter(self):
        parameter=[]
        for i in range(len(distribution)):
            layer_parameter={}
            for j in distribution[i].keys():
                if j=='b':
                    layer_parameter['b']=self.init_parameters_b(i)
                if j=='w':
                    layer_parameter['w']=self.init_parameters_w(i)
            parameter.append(layer_parameter)
        self.paraList=parameter
    def output(self,ary,pos):#forward propagation ary: std env Dict
        self.pos=pos
        ary=self.mod_env(ary)
        if(len(ary)!=dimensions[0]):
            throw_error("env input size incorrect to AI")
        v0_in=ary+self.paraList[0]['b']
        v0_out=activation[0](v0_in)
        v1_in=np.dot(v0_out,self.paraList[1]['w'])+self.paraList[1]['b']
        v1_out=activation[1](v1_in)
        maxstep,act_idx=0,-1#最大概率和对应选择
        flag=1
        while (flag==1):
            maxstep,act_idx=0,-1#最大概率和对应选择
            for i in range(len(v1_out)):
                if(v1_out[i]>maxstep):#FIXME 加入最大和次大选择的轮盘赌，增加随机性？
                    act_idx=i
                    maxstep=v1_out[i]
            if(-energy_cost[act_idx]<=ary[0]):#judge if energy is enough
                flag=0;
            v1_out[act_idx]=-1# 不足energy通过降低likelyhood保证AI换一个出
        self.lastact=self.currentact#v1.1 when make a choice,update it
        self.currentact=act_idx
        return act_idx
    def init_parameters_b(self,layer):
        dist=distribution[layer]['b']
        return np.random.rand(dimensions[layer])*(dist[1]-dist[0])+dist[0]
    def init_parameters_w(self,layer):
        dist=distribution[layer]['w']
        return np.random.rand(dimensions[layer-1],dimensions[layer])*(dist[1]-dist[0])+dist[0]
    def mod_env(self,env):#modify env from a dictionary to list(self-parameter first)
        ary=[]
        ary.append(env['energy'][self.pos])
        ary.append(env['energy'][1-self.pos])
        for i in range(OPTION_NUM):
            ary.append(env['step'][self.pos][i])
        for i in range(OPTION_NUM):
            ary.append(env['step'][1-self.pos][i])
        for i in range(OPTION_NUM):
            ary.append(env['stepp'][self.pos][i])
        for i in range(OPTION_NUM):
            ary.append(env['stepp'][1-self.pos][i])
        for i in range(OPTION_NUM):
            ary.append(env['steptot'][self.pos][i])
        for i in range(OPTION_NUM):
            ary.append(env['steptot'][1-self.pos][i])
        for i in range(OPTION_NUM):
            ary.append(env['stepnext'][self.pos][i])
        for i in range(OPTION_NUM):
            ary.append(env['stepnext'][1-self.pos][i])
        return ary
    def clear_database(self,degree):#v1.1
        if(degree>=1):
            self.database_ns = {0:[0,0,0,0,0,0],1:[0,0,0,0,0,0],2:[0,0,0,0,0,0],3:[0,0,0,0,0,0],4:[0,0,0,0,0,0],5:[0,0,0,0,0,0],-1:[0,0,0,0,0,0]}
            self.database_tot = [0,0,0,0,0,0]
        if(degree>=0):
            self.currentact=-1
            self.lastact=-1
        return
        
class CLS_Player(CLS_AI): #对AI类进行继承，以免出现版本推进导致函数不匹配问题，同时可以加入agentList，一视同仁
    def __init__(self,username):
        super(CLS_Player, self).__init__("Player "+username)#继承AI类的init方法
        self.username=username
    def output(self,ary,pos):
        player_action_chr=input("your choice:")
        player_action_num = actionDict[player_action_chr] #FIXME try,except for invalid choice
        while(1):
            if((player_action_num<=5) and (player_action_num>=0) and (ary['energy'][pos]>=-energy_cost[player_action_num])):#check energy sufficiency
                return player_action_num
            print("invalid choice")
            player_action_chr=input("choose again[e,k1,k2,k3,d1,d2]:")
            player_action_num = actionDict[player_action_chr] #FIXME try,except for invalid choice
            
class CLS_GameAgent(object):#an N times match between agents(At early version no difference between matches,later on added randomness)
    def __init__(self,agentList,agentDist):
        self.agentList=agentList
        self.env={
                "energy":[0,0],
                "step":[[0,0,0,0,0,0],[0,0,0,0,0,0]],
                "stepp":[[0,0,0,0,0,0],[0,0,0,0,0,0]],
                "steptot":[[0,0,0,0,0,0],[0,0,0,0,0,0]],
                "stepnext":[[0,0,0,0,0,0],[0,0,0,0,0,0]]
                 }
        self.agentDist=agentDist
    
    def game(self,agent1,agent2,count_score=1,echo=0):#一场游戏 count_score是否计分 echo是否输出详细战况
        rounds=0
        self.env_update(-1,-1,1,agent1,agent2)#env_init
        is_winner_1=0 #the winning state of agent1
        while(is_winner_1==0):
            rounds+=1
            if(rounds>=50):
                break
            act1=agent1.output(self.env,0)
            act2=agent2.output(self.env,1)
            is_winner_1,addpts=self.judge(act1,act2)
            agent1.pts+=addpts[0]*count_score
            agent2.pts+=addpts[1]*count_score
            self.env_update(act1,act2,0,agent1,agent2)
            if(echo):
                self.echoing(agent1,agent2,act1,act2,is_winner_1)
        agent1.pts+=is_winner_1*WINNING_PTS*count_score
        agent2.pts+=-1*is_winner_1*WINNING_PTS*count_score
        if(count_score==0):
            if(echo==0):
                is_winner_1 = (is_winner_1+1)/2
                #print("result:",is_winner_1)
            return is_winner_1
        return 
    #v1.11
    def contest(self,agent1,agent2,time):#v1.11 一场比赛，用于测试适应度
        self.env_update(-1,-1,2,agent1,agent2)#清空所有数据，AI初次见面
        while(time>0):
            time-=1
            self.game(agent1,agent2)
    def echoing(self,A1,A2,a1,a2,s):#Agent1 Agent2 act1 act2 status
        print(A1.family,"use action\"",actionDict_reverse[a1],"\"energy:",self.env["energy"][0])
        print(A2.family,"use action\"",actionDict_reverse[a2],"\"energy:",self.env["energy"][1])
        if(s==0):
            print("next round:")
        elif(s==1):
            print(A1.family,"kill",A2.family)
        elif(s==-1):
            print(A2.family,"kill",A1.family)
        return
    def judge(self,a1,a2):#judge winning(of player1)
        flag=judge_matrix[a1][a2]#win or not
        pts=(0,0)#FIXME 加入细致分数激励(pts_matrix)
        return flag,pts
    def env_update(self,act1,act2,init=0,agent1=0,agent2=0):# init: whether a new match has started,or a new competition started
        if(init==1):#new game
            self.env['energy'] = [0,0]
            self.env['step'] = [[0,0,0,0,0,0],[0,0,0,0,0,0]]
            self.env['stepp'] = [[0,0,0,0,0,0],[0,0,0,0,0,0]]
            if(agent1!=0 and agent2!=0):
                agent1.clear_database(0)
                agent2.clear_database(0)
            return
        elif(init==2):#new competition(several matches)
            self.env['energy'] = [0,0]
            self.env['step'] = [[0,0,0,0,0,0],[0,0,0,0,0,0]]
            self.env['stepp'] = [[0,0,0,0,0,0],[0,0,0,0,0,0]]
            self.env['steptot'] = [[0,0,0,0,0,0],[0,0,0,0,0,0]]
            self.env['stepnext'] = [[0,0,0,0,0,0],[0,0,0,0,0,0]]
            #v1.1
            if(agent1!=0 and agent2!=0):
                agent1.clear_database(1)
                agent2.clear_database(1)
            return
        #just update
        self.env['energy'][0]+=energy_cost[act1]# add/subtract energy
        self.env['energy'][1]+=energy_cost[act2]
        self.env['stepp']=self.env['step']# sec last step update
        self.env['step'] = [[0,0,0,0,0,0],[0,0,0,0,0,0]]
        self.env['step'][0][act1]=1
        self.env['step'][1][act2]=1
        #v1.1 style saving & update
        if(agent1!=0 and agent2!=0):
            agent1.database_ns[agent1.lastact][agent1.currentact]+=1
            agent1.database_tot[agent1.currentact]+=1
            agent2.database_ns[agent2.lastact][agent2.currentact]+=1
            agent2.database_tot[agent2.currentact]+=1
        self.env['stepnext'][0]=agent2.database_ns[agent2.currentact]#env 的【0】是agent1看到的agent2的信息
        self.env['stepnext'][1]=agent1.database_ns[agent1.currentact]
        self.env['steptot'][0]=agent2.database_tot
        self.env['steptot'][1]=agent1.database_tot
        return
    def print_data(self):
        for agent in agentList:
            print("agent",agent.family,",pts:",agent.pts)
    def test_agent(self,family,times=1):
        testee,test_score = self.agentDist[family],0
        for i in range(TEST_AI_NUM):
            family_icon = "Tester"
            tester_agent=CLS_AI(family_icon+str(i))
            tester_agent.init_parameter()
            print("with TestAgent",i,sep='',end=':\n')
            test_score=0
            for j in range(times):
                test_score+=self.game(testee,tester_agent,0)*100#不计分测试
            percentage=test_score/times
            print("agent",family,"scores",str(test_score)+"/"+str(times*100),str(percentage)+"%")#显示测试结果信息 eg: agent R0 scores 1300.0/2000 65.0%
    #v1.0 Player vs AI
    def PvA(self,player,family):
        opnt = self.agentDist[family]
        status=self.game(player,opnt,0,1)#不计分对战，player是玩家
        if(status==0):
            print(">>Tie<<")
        elif(status==1):
            print(">>Win<<")
        elif(status==-1):
            print(">>Lost<<")
    #v1.11
    def clear_agent_pts(self,clear_data=True):
        for agent in self.agentList:
            agent.pts=0
            if(clear_data):
                agent.clear_database(1)#simultaneously clear data

#------user data init-----
agentList=[]
agentDist={}
player = CLS_Player(input("username:")) #player是真人玩家
#GENERATION 1
for i in range(AI_NUM):
    #v1.11 support 2digits AI name(26*27=702 maximum)
    fnum=i+1
    family_icon,family_icon_2=chr(ord('A')-1+fnum%26),''
    if(ord(family_icon)<ord('A')):
        family_icon='Z'
    fnum = (fnum-1)//26
    if(fnum>0):
        family_icon_2 = chr(ord('A')-1+fnum)
        if(ord(family_icon_2)<ord('A')):
            family_icon_2='Z'
    
    temp_agent=CLS_AI(family_icon_2+family_icon)
    temp_agent.init_parameter()
    agentList.append(temp_agent)
    agentDist[temp_agent.family]=temp_agent
    print(agentList[i].family)
judge=CLS_GameAgent(agentList,agentDist)

for agent1 in agentList:
    for agent2 in agentList:
        if(agent1==agent2):
            continue
        judge.contest(agent1,agent2,1000)
        print(agent1.family,agent2.family,"done")
#judge.game(agentList[0],agentList[4])
judge.print_data()

#--------------------------------main----------------------------------
while True:
    order = input("List of Order:\n\t[t]for testing\n\t[c]for challenging\n\t[quit]to quit\norder:")
    if(order=='t'):
        agent_to_test=input("testing:")
        judge.test_agent(agent_to_test,5000)#random init 20(TEST_AI_NUM) AI to fight with the testers(1000 games v1.1)
    elif(order=='c'):
        agent_to_test=input("challenging:")
        judge.PvA(player,agent_to_test)
    elif(order=="quit"):
        break


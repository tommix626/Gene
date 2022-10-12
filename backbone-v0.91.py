# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 23:05:24 2021
version v0.91 
(2021年2月4日)-(2021年2月7日)

The true start of the project.
The backbone starts to set up all the structure without programming into details.

//log
    0.8     start of file
            the structure of GameAgent
            more completion on the structure map(in OneNote)
            communicate with zyc in txhy and take "style" into account(not implemented)
    0.9     finish the basic forward propagation step
            finish AI & GameAgent structural codes
            able to play with basic rules
    0.91    add testing for a specific agent,showing tangible results
            
    
//ideas
    -use numpy not pytorch
    -max rounds (finish)

//probs
    -go into deadcycles when running Gameagent.game function ,reason unclear(->solved)
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
AI_NUM=20 # the amount of AI playing
TEST_AI_NUM=20 # v0.91 another serveral agents for testing a player-choosed AI
ACTION=6  # the num of choice to act
judge_matrix=[[0,-1,0,0,-1,-1],[1,0,0,0,-1,-1],[0,0,0,0,-1,-1],[0,0,0,0,0,0],[1,1,1,0,0,-1],[1,1,1,0,1,0]]
                #given acts,decide the state of the first player
                #(judge_matrix[act1][act2]=1/0 means whether player 1 win this round)
energy_cost=[1,-1,0,-1,-2,-3]
actionDict={"e":0,"k1":1,"k2":4,"k3":5,"d1":2,"d2":3}   #for PvC use (->v1.0)
WINNING_PTS=100 #FIXME
#classes of objects
class CLS_AI(object):
    def __init__(self,family):
        self.family=family
        self.paraList=[]#dictionary with struct:[{'b':...},{'w':...,'b':...}]
        self.pts=0
        self.pos=-1#when playing the pos(0 or 1),which is used to indicate which parameter in the env is self-based
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
    def output(self,ary,pos):#forward propagation
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
    
    def game(self,agent1,agent2,count_score=1):#一场游戏 count_score是否计分
        rounds=0
        self.env_update(-1,-1,1)#env_init
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
            self.env_update(act1,act2)
        agent1.pts+=is_winner_1*WINNING_PTS*count_score
        agent2.pts+=-1*is_winner_1*WINNING_PTS*count_score
        if(count_score==0):
            print("result:",is_winner_1)
            is_winner_1 = (is_winner_1+1)/2
            return is_winner_1*100
        return 
    def judge(self,a1,a2):#judge winning(of player1)
        flag=judge_matrix[a1][a2]#win or not
        pts=(0,0)#FIXME 加入细致分数激励(pts_matrix)
        return flag,pts
    def env_update(self,act1,act2,init=0):# init: whether a new match has started,or a new competition started
        if(init==1):#new game
            self.env['energy'] = [0,0]
            self.env['step'] = [[0,0,0,0,0,0],[0,0,0,0,0,0]]
            self.env['stepp'] = [[0,0,0,0,0,0],[0,0,0,0,0,0]]
            return
        elif(init==2):#new competition(several matches)
            self.env['energy'] = [0,0]
            self.env['step'] = [[0,0,0,0,0,0],[0,0,0,0,0,0]]
            self.env['stepp'] = [[0,0,0,0,0,0],[0,0,0,0,0,0]]
            self.env['steptot'] = [[0,0,0,0,0,0],[0,0,0,0,0,0]]
            self.env['stepnext'] = [[0,0,0,0,0,0],[0,0,0,0,0,0]]
            return
        #just update
        self.env['energy'][0]+=energy_cost[act1]# add/subtract energy
        self.env['energy'][1]+=energy_cost[act2]
        self.env['stepp']=self.env['step']# sec last step update
        self.env['step'] = [[0,0,0,0,0,0],[0,0,0,0,0,0]]
        self.env['step'][0][act1]=1
        self.env['step'][1][act2]=1
        return
    def print_data(self):
        for agent in agentList:
            print("agent",agent.family,",pts:",agent.pts)
    def test_agent(self,family):
        testee,test_score = self.agentDist[family],0
        for i in range(TEST_AI_NUM):
            family_icon = "Tester"
            tester_agent=CLS_AI(family_icon+str(i))
            tester_agent.init_parameter()
            print("with TestAgent",i,sep='',end='')
            test_score+=self.game(testee,tester_agent,0)#不计分测试
        percentage=test_score/TEST_AI_NUM
        print("agent",family,"scores",str(test_score)+"/"+str(TEST_AI_NUM*100),str(percentage)+"%")#显示测试结果信息 eg: agent R0 scores 1300.0/2000 65.0%

#------user data init-----
agentList=[]
agentDist={}
for i in range(AI_NUM):
    family_icon = chr(ord('A')+i)
    temp_agent=CLS_AI(family_icon+'0')
    temp_agent.init_parameter()
    agentList.append(temp_agent)
    agentDist[temp_agent.family]=temp_agent
    print(agentList[i].family)
judge=CLS_GameAgent(agentList,agentDist)
for agent1 in agentList:
    for agent2 in agentList:
        if(agent1==agent2):
            continue
        judge.game(agent1,agent2)
        print(agent1.family,agent2.family,"done")
#judge.game(agentList[0],agentList[4])
judge.print_data()
#--------------------------------main----------------------------------
while True:
    agent_to_test=input("testing:")
    judge.test_agent(agent_to_test)


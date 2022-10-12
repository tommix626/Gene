# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 23:05:24 2021
version v0.8 (unrunnable)
The true start of the project.
The backbone starts to set up all the structure without programming into details.

//log
    0.8     start of file
            the structure of GameAgent
            more completion on the structure map(in OneNote)
            communicate with zyc in txhy and take "style" into account
    
//ideas
    -use numpy not pytorch


@author: Tom
"""
import sys,pygame
import numpy as np


#global variables&consts
dimensions=[38,6]
distribution=[
    {'w':[-1,1]},{'b':[-1,1]}#FIXME 
]
AI_NUM=10 # the amount of AI playing

#classes of objects
class CLS_AI(object):
    def __init__(self,family):
        self.family=family
        self.wList=[]#size=(38)
        self.pts=0
    def init_parameter(self):
        parameter=[]
        for i in range(len(distribution)):
            layer_parameter={}
            for j in distribution[i].keys():
                if j=='b':
                    layer_parameter['b']=init_parameters_b(i)
                if j=='w':
                    layer_parameter['w']=init_parameters_w(i)
            parameter.append(layer_parameter)
        self.wList=parameter
    def output(self,ary):#forward propagation
        np.dot(ary,wList['w'])+wList['b']
        

        
class CLS_GameAgent(object):#an N times match between agents(At early version no difference between matches,later on added randomness)
    def __init__(self,agentList):
        self.agentList=agentList
        self.env=[0]*38#存放38个变量的status
    
    def game(self,agent1,agent2):#一场游戏
        pts=[0,0]
        is_winner=0 #whether the game is over
        while(is_winner==0):
            act1=agent1.output(self.env)
            act2=agent2.output(self.env)
            is_winner,addpts=self.judge(act1,act2)
            agent1.pts+=addpts[0]
            agent2.pts+=addpts[1]
            self.env_update()
        return pts;
    def judge(self,a1,a2):#judge winning
        flag=0#win or not
        if(获胜条件):
            flag=1
        pts=(0,0)#FIXME 加入细致分数激励
        return flag,pts
    def env_update(self):
        
        return


#------user data init-----
agentList=[CLS_AI("A")]*AI_NUM
#--------------------------------main----------------------------------


# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 22:23:37 2021

@author: Tom
"""


def f(a):
    def g(x):
        return 3*x
    return 2*g(a)
#print(f(2))
    
class a(object):
    def __init__(self,val):
        self.val=val
def cmp(x):
    return x.val
aList=[]
for i in range(10):
    temp = a(i)
    aList.append(temp)
aList.sort(key=cmp,reverse=1)
for i in range(10):
    print(aList[i].val)
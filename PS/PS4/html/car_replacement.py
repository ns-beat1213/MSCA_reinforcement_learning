#!/usr/bin/env python
# coding: utf-8

# ## Home Work 4
# Machine replacement problem

# Shogo Nakano

# ### Setting

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


# read table from csv
df = pd.read_csv('C:/Users/nsbea/OneDrive/5_core/RL/code/howard_autoReplacement_data.csv')  


# In[3]:


df.head()


# C: cost of buying car of age i\
# T: trade in value of age i \
# E cost of operations in age i \
# P: survival probability

# In[4]:


# generate list from 1 to 41
action = np.arange(1, 42, 1)
action.shape


# ### transiton probability

# In[5]:


# Generate 40 * 40 matrix for transition probability
# Transition probability matrix
P = np.zeros((len(action), 40, 40))
_, s, _ = P.shape

for a in action:
    P_mat = np.zeros((40, 40))
    
    # If a == 1, keep the car
    if a == 1:
        for i in range(s):
            for j in range(s):
                if i+1 == j:
                    P_mat[i, j] = df['p'][i+1]
                if j == s-1:
                    P_mat[i, j] = 1 - df['p'][i+1]
        P_mat[38, 39] = 1
    
    # If a is 41, the probability is 1 (stay terminal state)
    elif a == 41:
        for i in range(s):
            P_mat[i, -1] = 1
    
    # For other actions,
    else:
        for i in range(s):
            P_mat[i, a-2] = df['p'][a-2]
            P_mat[i, -1] = 1 - df['p'][a-2]
            
    P[a-1] = P_mat


# ### Reward

# If keep the car
# - the reward is the maintain cost
# 
# If buy a car
# - the reward is Trade value - buying cost - maintain cost
# 

# In[6]:


# define reward matrix
q = np.zeros((41,40,1),dtype=np.float64)

#K is the number of actions
#N is the number of states
K,N,M = P.shape

# reward matrix
for i in range(0,N):
    for k in range(0,K):
        if k == 0:
            q[k,i,0] = -df['E'][i+1]
        else:
            q[k,i,0] = df['T'][i+1] - df['C'][k-1] - df['E'][k-1]


# ### Policy iteration

# In[7]:


# policy iteration

T=8

# initiate objects to store value function and decisions
v = np.zeros((N,T),dtype=np.float64)
d = np.ones((N,T),dtype=int)

# initiate auxiliary variables
PP = np.zeros((N,N),dtype=np.float64)
qvec = np.zeros((N,1),dtype=np.float64)

# policy iteration
for n in range(1,T):
    # policy improvement (assume v=0)
    for i in range(0,N):
        rhs = np.zeros((1,K),dtype=np.float64)
        for k in range(0,K):
            rhs[0][k] = q[k][i][0] + np.matmul(P[k,i,:],v[:,n-1]) 
        v[i,n] = max(rhs[0])
        d[i,n] = np.argmax(rhs[0])

    # value determination
    for i in range(0,N):
        PP[i,:] = P[d[i,n],i,:]
    
    A = np.concatenate((np.identity(N)-PP,np.ones((N,1))),axis=1)
    A = np.delete(A,N-1,1)
    
    for i in range(0,N):
        qvec[i,0] = q[d[i,n],i,0]

    tmp = np.matmul(np.linalg.inv(A),qvec)
    g = tmp[N-1]
    tmp[N-1] = 0
    # iteration
    v[:,n] = tmp.T

v = v.T
d = d.T


# ### Result

# In[8]:


# get action from last iter and convert index to action
decision = np.where(d[-1] == 0, 'K', d[-1] - 1)

# create dataframe
data = {'Decision': decision, 'Value': v[-1]}
result = pd.DataFrame(data)

# change index
new_index = list(range(1, 41))
result.index = new_index[:len(df)]

# show
result


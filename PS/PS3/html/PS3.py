#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ## 1

# ### 1.1 transition probability matrix

# In[282]:


# transition matrix 
# Initialize the transition probability matrix
P_a = np.zeros((3, 15, 15))

# Probability matrix for e to e'
Pe = np.array([[0.8, 0.1, 0.1],
               [0.01, 0.98, 0.01],
               [0.1, 0.1, 0.8]])

# Iterate through all possible values for s, e, a
for s in range(5):
    for e in range(3):
        for a in range(3):
            # Calculate s' for all possible values of e'
            for e_prime in range(3):
                s_prime = min(s,a) + e_prime
                if 0 <= s_prime <= 4:
                    index_from = 3 * s + e
                    index_to = 3 * s_prime + e_prime
                    P_a[a, index_from, index_to] = Pe[e, e_prime]


# In[245]:


map = {0:(0,0), 1:(0,1), 2:(0,2), 3:(1,0), 4:(1,1), 5:(1,2), 6:(2,0), 7:(2,1), 8:(2,2), 9:(3,0), 10:(3,1), 11:(3,2), 12:(4,0), 13:(4,1), 14:(4,2)}


# In[246]:


# plot three heatmaps for each action
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for a in range(3):
    ax = axes[a]
    ax.set_title(rf"$P_{{(s,e),(s' ,e')}}^{a}$")
    ax.set_xlabel('state')
    ax.set_ylabel('next state')
    sns.heatmap(P_a[a], ax=ax, cmap='Blues', annot=True, annot_kws={'fontsize': 'x-small'}, xticklabels=list(map.values()), yticklabels=list(map.values()))
fig.tight_layout(rect=[0, 0.1, 1, 1])


# ### 1.2 reward matrix

# In[247]:


# define utility
def utility(c):
    return np.log(c+1) if c >=0 else 0 # use zero for negative consumption


# In[248]:


# set zero matrix for reward
R = np.zeros((3, 15, 15), dtype=np.float64)

# Iterate through all possible values for s, e, a
for s in range(5):
    for e in range(3):
        for a in range(3):
            # Calculate utility for all possible values of e'
            for e_prime in range(3):
                s_prime = min(s,a) + e_prime
                c = s - min(s, a)
                r = utility(c)
                if 0 <= s_prime <= 4:
                    index_from = 3 * s + e
                    index_to = 3 * s_prime + e_prime
                    R[a, index_from, index_to] = r


# In[249]:


# plot three heatmaps for each action
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for a in range(3):
    ax = axes[a]
    sns.heatmap(R[a], ax=ax, cmap='Blues', annot=True, annot_kws={'fontsize': 'x-small'}, xticklabels=list(map.values()), yticklabels=list(map.values()))
    ax.set_xlabel('next state')
    ax.set_ylabel('state')
    ax.set_title(rf"$R_{{(s,e),(s' ,e')}}^{a}$")
fig.tight_layout()


# ### 1.3 expected utility

# In[250]:


# function for expcted reward
def get_expected_reward(q, num_action, num_state, horizon, discount=0.95):
    # store the value function for each iteration
    v = np.zeros((horizon+1, num_action, num_state), dtype=np.float64)
    # t = iteration number
    for t in range(1, horizon+1):
        # k = action
        for k in range(num_action):
            # i = state
            for i in range(num_state):
                # calculate the expected reward
                v[t, k, i] = q[k,i] + discount * np.matmul(P_a[k, i, :], v[t-1,k])

    # return the value function at the last iteration
    return v[-1], horizon


# In[251]:


# Calculate the expected immediate reward
expected_reward = np.sum(P_a * R, axis=-1)

# Reshape the expected reward to have dimensions 2x21x1
expected_reward = expected_reward.reshape(3, -1, 1)


# In[253]:


# get value function
v, _ = get_expected_reward(q=expected_reward, num_action=3, num_state=15, horizon=600, discount=0.95)


# In[254]:


# plot value function
fig, ax = plt.subplots(1, 1, figsize=(5, 5))
ax.set_title(rf"$V^{{*}}$")

# Get maximum value indexes in each row
max_indices = np.argmax(v, axis=0)

# Plot the heatmap
heatmap = sns.heatmap(v.T, ax=ax, cmap='Blues', annot=True, annot_kws={'fontsize': 'x-small'}, fmt='.2f', yticklabels=list(map.values()))


# Emphasize the maximum values
for t, max_index in enumerate(max_indices):
    heatmap.texts[t * v.shape[0] + max_index].set_weight('bold')
    heatmap.texts[t * v.shape[0] + max_index].set_color('red')
#fig.tight_layout(rect=[0, 0.1, 1, 1])
plt.xlabel('action')
plt.ylabel('state')
plt.show()


# In[257]:


# save value function from value iteration
v_vi = v[0]


# ### 1.4 plot value function

# In[258]:


# line plot for value function
fig, ax = plt.subplots(1, 1, figsize=(5, 5))
ax.set_title(rf"$V^{{*}}$")
ax.set_xlabel('s')
ax.set_ylabel('value')
for e in range(3):
    # get slice to plot
    ax.plot(np.max(v, axis=0)[e::3], label=f'e={e}')
ax.legend()
plt.xticks(np.arange(0, 5, 1))
plt.show()


# ### 1.5 plot optimal policy

# In[259]:


# line plot for optimal policy
fig, ax = plt.subplots(1, 1, figsize=(5, 5))
ax.set_title(rf"$a^{{*}}$")
ax.set_xlabel('s')
ax.set_ylabel('a')
for e in range(3):
    ax.plot(np.argmax(v, axis=0)[e::3], label=f'e={e}')
ax.legend()
plt.xticks(np.arange(0, 5, 1))
plt.show()


# Optimal policy is always a=0.

# ### 1.6 plot optimal consuming

# In[260]:


# line plot for optimal policy
fig, ax = plt.subplots(1, 1, figsize=(5, 5))
ax.set_title(rf"$C^{{*}}$")
ax.set_xlabel('s')
ax.set_ylabel('c')

# set save vector
s_vector = [0, 1, 2, 3, 4]
for e in range(3):
    # get optimal action
    a_vector = np.argmax(v, axis=0)[e::3]
    # calculate consumption (s - a)
    c_vector = s_vector - a_vector
    ax.plot(c_vector, label=f'e={e}')
ax.legend()
plt.xticks(np.arange(0, 5, 1))
plt.show()


# optimal a =0, this means always s = a (comsume everything)

# ### 1.7 simulation

# In[261]:


import pandas as pd
import numpy as np

def simulate_se(s, e):
    """""
    Simulate the state transition of (s, e) for 1000 periods

    Parameters
    ----------
    s : int
        initial s
    e : int
        initial e

    Returns
    -------
    df : pd.DataFrame
        dataframe with columns t, s, e, s_prime, e_prime, reward
    """""
    # get index of state
    index = 3 * s + e

    # optimal policy is always choose a = 0, so get p_vector and r_vector
    p_opt = P_a[0, index, :]
    r_opt = R[0, index, :]

    # empty lists to store the results
    list_t = []
    list_s = []
    list_e = []
    list_s_prime = []
    list_e_prime = []
    list_reward = []

    for t in range(1000):
        # get next state
        next_index = np.random.choice(len(p_opt), p=p_opt)
        # get reward
        reward = R[0, index, next_index]
        # get s and e
        s_prime = next_index // 3
        e_prime = next_index % 3

        # save the result
        list_t.append(t)
        list_s.append(s)
        list_e.append(e)
        list_s_prime.append(s_prime)
        list_e_prime.append(e_prime)
        list_reward.append(reward)

        # get p_vector and r_vector
        p_opt = P_a[0, next_index, :]
        r_opt = R[0, next_index, :]

        # one step ahead
        s = s_prime
        e = e_prime
        index = next_index

    # dataframe to store the results
    df = pd.DataFrame({'t': list_t,
                       's': list_s,
                       'e': list_e,
                       's\'': list_s_prime,
                       'e\'': list_e_prime,
                       'r': list_reward
                       })
    df['beta'] = 0.95
    df['discount_reward'] = df['r'] * df['beta'] ** df['t']
    df['a'] = 0
    df['c'] = df['s'] - df['a']
    
    return df



# In[262]:


# get result from sample run
df_sim = simulate_se(s=0, e=0)


# In[263]:


# plot a c, and s
fig, ax = plt.subplots(1, 1, figsize=(10, 3))
ax.set_title(rf"$a, c, s$")
ax.set_xlabel('t')
ax.set_ylabel('value')
ax.set_xlim(0, 100)
ax.plot(df_sim['a'], label='a')
ax.plot(df_sim['c'], label='c')
ax.plot(df_sim['s'], label='s')
ax.legend()
plt.show()


# ### 1.8 transiton matrix and reward matrix which yield highest expected rewards

# Optimal policy is always choose a = 0.

# In[264]:


Pmat = P_a[0, :, :]
Rmat = R[0, :, :]

# two heatmaps for P and R
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].set_title(rf"$P$")
ax[1].set_title(rf"$R$")
sns.heatmap(Pmat, ax=ax[0], cmap='Blues', annot=True, annot_kws={'fontsize': 5}, fmt='.2f', yticklabels=list(map.values()), xticklabels=list(map.values()))
sns.heatmap(Rmat, ax=ax[1], cmap='Blues', annot=True, annot_kws={'fontsize': 5}, fmt='.2f', yticklabels=list(map.values()), xticklabels=list(map.values()))
ax[0].set_xlabel('s\'')
ax[0].set_ylabel('s')
ax[1].set_xlabel('s\'')
ax[1].set_ylabel('s')
fig.tight_layout()
plt.show()


# ### 1.9 Calculate the value function for the Markok process with rewards

# In[265]:


# calculate the value function from Pmat and Rmat
T=1000
N,M = Rmat.shape
discount = 0.95

q = np.zeros((N,1),dtype=np.float64)
for i in range(0,N):
    for j in range(0,N):
        q[i] += np.multiply(Rmat[i,j],Pmat[i,j])

v = np.zeros((N,T),dtype=np.float64)

for n in range(1,T):
    for i in range(0,N):
        v[i,n] = q[i] + discount * np.matmul(Pmat[i,:],v[:,n-1])

v_simulation = v[:,-1]


# In[278]:


# compare the value function from value iteration and simulation, bar plot
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
ax.set_title(rf"$V^{{*}}$")
ax.set_xlabel('s')
ax.set_ylabel('value')

bar_width = 0.4  # Set the bar width
positions = np.arange(0, 15, 1)

ax.bar(positions - bar_width / 2, v_simulation, bar_width, label='simulation', color='lightcoral')
ax.bar(positions + bar_width / 2, v_vi, bar_width, label='value iteration', color='lightskyblue')

# Set xticks and xticklabels
ax.set_xticks(positions)
ax.set_xticklabels(list(map.values()))

ax.legend()
plt.show()


# It matchs the previous value function.

# ### 1.10 Simulate the Markov process with rewards from the previous question for starting at each state pair 

# In[279]:


def simulate_multiple(s, e, n):
    """""
    Simulate the state transition of (s, e) for 1000 periods n times

    Parameters
    ----------
    s : int
        initial s
    e : int
        initial e
    n : int
        number of simulations

    Returns
    -------
    rewards : list
        list of rewards
    """""
    rewards = []
    for i in range(n):
        df_sim = simulate_se(s, e)
        # get sum of discount reward
        sim_reward = df_sim['discount_reward'].sum()
        rewards.append(sim_reward)
    return rewards


# In[238]:


# simulate each initial state 1000 times

# dictionary to store the results
dict_results = {}

for index in range(15):

    # get s and e from index
    s = index // 3
    e = index % 3

    # run simulation, temp contains 1000 simulated rewards given s and e
    temp = simulate_multiple(s, e, n=1000)

    # save the results
    dict_results[index] = temp

# convert to dataframe
df_results = pd.DataFrame(dict_results)
    


# In[280]:


# average of each column
average_reward = df_results.mean(axis=0)


# In[281]:


# compare the value function from value iteration and simulation, bar plot
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
ax.set_title(rf"$V^{{*}}$")
ax.set_xlabel('s')
ax.set_ylabel('value')

bar_width = 0.4  # Set the bar width
positions = np.arange(0, 15, 1)

ax.bar(positions - bar_width / 2, average_reward, bar_width, label='simulation', color='lightcoral')
ax.bar(positions + bar_width / 2, v_vi, bar_width, label='value iteration', color='lightskyblue')

# Set xticks and xticklabels
ax.set_xticks(positions)
ax.set_xticklabels(list(map.values()))

ax.legend()
plt.show()


# It matchs the previous value function.

# ### 1.11 Policy iteration

# In[304]:


# define transition probability and reward matrix for each action
P = P_a.copy()
RR= R.copy()

K,N,M = P.shape

# compute expected reward for each state
q = np.zeros((K,N,1),dtype=np.float64)

for i in range(0,N):
    for k in range(0,K):
        for j in range(0,M):
            q[k,i,0] = q[k,i,0] + RR[k,i,j]*P[k,i,j]
T=100

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


# In[306]:


d[-1,:]


# The result is different from value iteration and simulation... maybe need to fix

# ## 2

# ### 2.1 trainsition probability matrix

# In[2]:


# get transition matrix right
transition_matrix_right = np.zeros((21,21))

for i in range(1,20):
    transition_matrix_right[i,i-1] = 0.2
    transition_matrix_right[i,i+1] = 0.8

transition_matrix_right[0,0] = 1
transition_matrix_right[20,20] = 1

# get transition matrix left
transition_matrix_left = np.zeros((21,21))

for i in range(1,20):
    transition_matrix_left[i,i-1] = 0.8
    transition_matrix_left[i,i+1] = 0.2

transition_matrix_left[0,0] = 1
transition_matrix_left[20,20] = 1

# combine transition matrix
transition_matrix = np.zeros((2, 21,21))
transition_matrix[0] = transition_matrix_left
transition_matrix[1] = transition_matrix_right


# In[3]:


# plot the heatmap of transition matrix
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].set_title('transition matrix left')
ax[1].set_title('transition matrix right')
sns.heatmap(transition_matrix[0], ax=ax[0], cmap='Blues', annot=True, annot_kws={'fontsize': 'x-small'})
sns.heatmap(transition_matrix[1], ax=ax[1], cmap='Blues', annot=True, annot_kws={'fontsize': 'x-small'})
fig.tight_layout(rect=[0, 0.1, 1, 1])
plt.show()


# ### 2. reward matrix

# In[4]:


# get transition matrix right
reward_matrix_right = np.zeros((21,21))

for i in range(1,20):
    reward_matrix_right[i,i-1] = 0.05
    reward_matrix_right[i,i+1] = 0.05

reward_matrix_right[0,0] = 0
reward_matrix_right[20,20] = 0
reward_matrix_right[19,20] = 1
reward_matrix_right[1,0] = -1

# get transition matrix left
reward_matrix_left = reward_matrix_right.copy()

# combine transition matrix
reward_matrix = np.zeros((2, 21,21))
reward_matrix[0] = reward_matrix_left
reward_matrix[1] = reward_matrix_right


# In[6]:


# plot the heatmap of reward matrix
pal = sns.diverging_palette(220, 20, as_cmap=True)

fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].set_title('reward matrix left')
ax[1].set_title('reward matrix right')
sns.heatmap(reward_matrix[0], ax=ax[0], cmap=pal, annot=True, annot_kws={'fontsize': 'x-small'})
sns.heatmap(reward_matrix[1], ax=ax[1], cmap=pal, annot=True, annot_kws={'fontsize': 'x-small'})
fig.tight_layout(rect=[0, 0.1, 1, 1])
plt.show()


# ### 3. expected immedicate reward

# In[43]:


n_action = 2
n_state = 21

# set q
q = np.zeros((n_action, n_state), dtype=np.float64)

# compute expected reward for each state
for a in range(0,n_action):
    q[a] = np.sum(transition_matrix[a] * reward_matrix[a], axis=1).reshape(n_state)


# In[8]:


def get_expected_reward(q, horizon):
    # store expected reward
    v = np.zeros((horizon+1, n_action, n_state), dtype=np.float64)
    
    # compute expected reward for each state
    for t in range(1, horizon+1):
        for a in range(n_action):
            for i in range(n_state):
                v[t, a, i] = q[a, i] + np.matmul(transition_matrix[a, i, :], v[t-1, a])
    # get last state
    return v[-1], horizon


# In[9]:


# get immediate reward
immediate_reward, _ = get_expected_reward(q, 1)


# In[45]:


# heatmap of immediate reward
fig, ax = plt.subplots(1, 1, figsize=(6, 6))
ax.set_title('immediate reward')
sns.heatmap(immediate_reward.T[1:20,:], ax=ax, cmap=pal, annot=True, annot_kws={'fontsize': 'x-small'})
ax.set_yticklabels(np.arange(1, 20))
fig.tight_layout(rect=[0, 0.1, 1, 1])
plt.show()


# ### 2-4 2-5. matrix inversion technique

# In[61]:


# always go right
policy = 1

# Get the transition probability matrix (P) and immediate reward matrix (q) for the policy
P_policy = transition_matrix[policy]
q_policy = immediate_reward[policy]

# Calculate the state value function V using the matrix inversion technique
X_right = np.squeeze(
        np.matmul(
            np.linalg.inv(np.eye(n_state-2) - P_policy[1:-1, 1:-1]),
            q_policy[1:-1, np.newaxis]
        ))


# In[63]:


# akways go left
policy = 0

# Get the transition probability matrix (P) and immediate reward matrix (q) for the policy
P_policy = transition_matrix[policy]
q_policy = immediate_reward[policy]

# Calculate the state value function V using the matrix inversion technique
X_left = np.squeeze(
    np.matmul(
        np.linalg.inv(np.eye(n_state-2) - P_policy[1:-1, 1:-1]),
        q_policy[1:-1, np.newaxis]
    ))


# In[70]:


# heatmap of value function
fig, ax = plt.subplots(1, 1, figsize=(6, 6))
ax.set_title('value function')
sns.heatmap(np.vstack((X_left, X_right)).T[0:20,:], ax=ax, cmap=pal, annot=True, annot_kws={'fontsize': 'x-small'})
ax.set_yticklabels(np.arange(1, 20))
fig.tight_layout(rect=[0, 0.1, 1, 1])
plt.show()


# ### 2-6 2-7. Evaluate the  policy using the recursive equation/iteration

# In[75]:


# get immediate reward
reward, _ = get_expected_reward(q, 100)


# In[76]:


# heatmap of value function
fig, ax = plt.subplots(1, 1, figsize=(6, 6))
ax.set_title('value function')
sns.heatmap(reward.T[1:20,:], ax=ax, cmap=pal, annot=True, annot_kws={'fontsize': 'x-small'})
ax.set_yticklabels(np.arange(1, 20))
fig.tight_layout(rect=[0, 0.1, 1, 1])
plt.show()


# ### 2-8. Find the optimal policy and the value function associated with the optimal policy

# In[112]:


def get_optimal_policy(q, horizon, n_action, n_state, transition_matrix):
    # store expected reward
    v = np.zeros((horizon+1, n_action, n_state), dtype=np.float64)
    value = np.zeros((horizon+1, n_state), dtype=np.float64)
    d = np.zeros((horizon+1, n_state), dtype=np.int32)

    # compute expected reward for each state
    for t in range(1, horizon+1):
        for a in range(n_action):
            for i in range(n_state):
                v[t, a, i] = q[a, i] + np.matmul(transition_matrix[a, i, :], v[t-1, a]) *0.95

        value[t] = np.max(v[t], axis=0)
        d[t] = np.argmax(v[t], axis=0)

    # get last state
    return value, d, horizon


# In[114]:


value, d, _ = get_optimal_policy(q, 600, 2, 21, transition_matrix)


# In[117]:


# optimal policy
d[-1]


# Always right policy

# In[118]:


# value function
v[-1]


# Not much the previous question, why?

# ### 2-9. use finitehorizon function

# In[36]:


import mdptoolbox.mdp as mdp


# In[136]:


# finite holizon
fh = mdp.FiniteHorizon(transition_matrix, reward_matrix, 0.95, 100)
fh.run()

V = fh.V
policy = fh.policy


# In[139]:


# optimal policy
policy[1:20, 0]


# always right policy

# In[140]:


# value function
V[1:20, 0]


# ### 2-10. use valueiteration function

# In[141]:


# value iteration
vi = mdp.ValueIteration(transition_matrix, reward_matrix, 0.95)
vi.run()


# In[142]:


# optimal policy
vi.policy


# This is always right policy

# In[143]:


# value function
vi.V


# This value is same as 2-9.

# ### 2-11 Is the “always go right” policy the optimal policy?

# I think so. The value iteration and matrix inversion shows that the robot should choose "go right" in all state.

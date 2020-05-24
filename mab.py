# Multi-Armed Bandit 
import numpy as np
from scipy import stats # import beta, bernoulli
import matplotlib.pyplot as plt
from numba import jit
import numba as nb
import seaborn as sns
import pandas as pd

def sns_lineplot(values_2D):
    N_repeats, N_episodes = values_2D.shape
    time = np.tile(np.arange(N_episodes), N_repeats)
    repeats = np.repeat(np.arange(N_repeats), N_episodes)
    data = np.concatenate([time.reshape(-1,1), values_2D.reshape(-1,1), repeats.reshape(-1,1)],1)
    df = pd.DataFrame(data=data, columns=['time', 'regrets', 'repeats'])
    sns.relplot(x='time',y='regrets',data=df,kind='line')


def sns_lineplot_dic(values_2D_dic, T_interval=1):
    key_list = []
    df_list = []
    for idx, key in enumerate(values_2D_dic.keys()):
        N_repeats, N_episodes = values_2D_dic[key].shape
        time = np.tile(np.arange(N_episodes)*T_interval, N_repeats)
        repeats = np.repeat(np.arange(N_repeats), N_episodes)
        data_init = np.concatenate([time.reshape(-1,1), repeats.reshape(-1,1), values_2D_dic[key].reshape(-1,1)],1)
        df = pd.DataFrame(data=data_init, columns=['time', 'repeats', 'regrets'])
        df['label'] = np.repeat([key], N_repeats * N_episodes)
        df_list.append(df)
    df = pd.concat(df_list)            
    sns.relplot(x='time',y='regrets',hue='label',data=df,kind='line')


def mab_bernoulli_ts_plot(max_action_a, regrets_a):
    plt.plot(max_action_a)
    plt.show()
    plt.semilogx(regrets_a)
    plt.show()


def mab_bernoulli_ts_run(mu_a, N_actions=10, N_episodes=100):
    s0, f0 = 1, 1
    Si, Fi = np.zeros(N_actions,dtype=int), np.zeros(N_actions,dtype=int)
    theta = np.zeros(N_actions,dtype=float)
    max_action_a = np.zeros(N_episodes,dtype=int)
    
    for e in range(N_episodes):
        for action in range(N_actions):
            theta[action] = stats.beta.rvs(Si[action]+s0, Fi[action]+f0)
        max_action = np.argmax(theta)
        reward = stats.bernoulli.rvs(mu_a[max_action])
        Si[max_action] += reward 
        Fi[max_action] += 1 - reward
        max_action_a[e] = max_action

    return max_action_a


def mab_bernoulli_ts_episodes(mu_a, N_actions=10, N_episodes=100):
    max_action_a = mab_bernoulli_ts_run(mu_a, N_actions=N_actions, N_episodes=N_episodes)
    regrets = 0
    regrets_a = np.zeros(N_episodes,dtype=float)
    for e in range(N_episodes):
        regrets += np.max(mu_a) - mu_a[max_action_a[e]]
        regrets_a[e] = regrets
    return regrets_a

def mab_bernoulli_fn_episodes(mu_a, N_actions=10, N_episodes=100, fn_run=mab_bernoulli_ts_run):
    max_action_a = fn_run(mu_a, N_actions=N_actions, N_episodes=N_episodes)
    regrets = 0
    regrets_a = np.zeros(N_episodes,dtype=float)
    for e in range(N_episodes):
        regrets += np.max(mu_a) - mu_a[max_action_a[e]]
        regrets_a[e] = regrets
    return regrets_a    

@jit('float64(float64,float64)')
def KL(p, q):
    return p*np.log(p/q) + (1-p)*np.log((1-p)/(1-q))

def mab_low_bound_one(mu_a):
    N_actions = mu_a.shape[0]
    mu_a = np.sort(mu_a)[::-1]
    exp_regrets = 0.0
    for action in range(1, N_actions): # assume mu_a[0] = npmax(mu_a)
        exp_regrets += (mu_a[0] - mu_a[action]) / KL(mu_a[action], mu_a[0])
    return exp_regrets


def mab_bernoulli_low_bound(mu_a, N_actions=10, N_episodes=100):
    exp_regrets = 0.0
    for action in range(1, N_actions): # assume mu_a[0] = npmax(mu_a)
        exp_regrets += (mu_a[0] - mu_a[action]) / KL(mu_a[action], mu_a[0])
    exp_regrets_a = np.zeros(N_episodes,dtype=float)
    exp_regrets_a[1:] = np.log(range(1, N_episodes)) * exp_regrets
    return exp_regrets_a


def mab_bernoulli_greedy_run(mu_a, N_actions=10, N_episodes=100):
    max_action_a = np.zeros(N_episodes,dtype=int)    
    q_val = np.zeros(N_actions,dtype=float)
    n_action = np.zeros(N_actions,dtype=int)

    for e in range(N_episodes):
        if e == 0:
            max_action = np.random.randint(N_actions)    
        else:
            max_action = np.argmax(q_val)
        reward = stats.bernoulli.rvs(mu_a[max_action])
        n_action[max_action] += 1
        q_val[max_action] = (q_val[max_action] * (n_action[max_action]-1) + reward) / n_action[max_action]
        max_action_a[e] = max_action

    return max_action_a


def mab_bernoulli_ts_episodes_repeats(mu_a, N_actions=10, N_episodes=100, N_repeats=10):
    regrets_a_ts_repeat = np.zeros((N_repeats, N_episodes),dtype=float)

    for rp in range(N_repeats):
        regrets_a_ts = mab_bernoulli_ts_episodes(mu_a, N_actions=N_actions, N_episodes=N_episodes)
        regrets_a_ts_repeat[rp] = regrets_a_ts

    return regrets_a_ts_repeat


def mab_bernoulli_fn_episodes_repeats(mu_a, N_actions=10, N_episodes=100, N_repeats=10, fn_run=mab_bernoulli_ts_run):
    regrets_a_ts_repeat = np.zeros((N_repeats, N_episodes),dtype=float)

    for rp in range(N_repeats):
        regrets_a_ts = mab_bernoulli_fn_episodes(mu_a, N_actions=N_actions, N_episodes=N_episodes, fn_run=fn_run)
        regrets_a_ts_repeat[rp] = regrets_a_ts

    return regrets_a_ts_repeat



def mab_bernoulli_ts(plot_type='seaborn'):
    """
    mab_bernoulli_ts()

    Multi-armed bandit (MAB) system with Bernoulli reward optimized by Thompson sampling (TS)
    """
    print('Hello, MAB')
    N_actions=10
    N_episodes=1000
    N_repeats = 10
    epsilon=0.1
    mu_a = np.zeros(N_actions, dtype=float) + 0.5 - epsilon
    mu_a[0] = 0.5
    regrets_a_ts_repeat = mab_bernoulli_ts_episodes_repeats(mu_a, N_actions=N_actions, N_episodes=N_episodes, N_repeats=N_repeats)
    regrets_a_lb = mab_bernoulli_low_bound(mu_a, N_actions=N_actions, N_episodes=N_episodes)

    if plot_type == 'seaborn':
        sns_lineplot(regrets_a_ts_repeat)
    else:
        plt.semilogx(np.mean(regrets_a_ts_repeat,axis=0), label='TS')
        plt.semilogx(regrets_a_lb, label='LB')
        plt.xlabel('T')
        plt.ylabel('regrets')
        plt.legend(loc=0)
        plt.show()
    plt.show()

def mab_bernoulli_fn(plot_type='seaborn'):
    """
    mab_bernoulli_ts()

    Multi-armed bandit (MAB) system with Bernoulli reward optimized by Thompson sampling (TS)
    """
    print('Hello, MAB')
    N_actions=10
    N_episodes=1000
    N_repeats = 10
    epsilon=0.1
    mu_a = np.zeros(N_actions, dtype=float) + 0.5 - epsilon
    mu_a[0] = 0.5
    regrets_a_ts_repeat = mab_bernoulli_fn_episodes_repeats(mu_a, N_actions=N_actions, N_episodes=N_episodes, N_repeats=N_repeats, fn_run=mab_bernoulli_ts_run)
    regrets_a_greedy_repeat = mab_bernoulli_fn_episodes_repeats(mu_a, N_actions=N_actions, N_episodes=N_episodes, N_repeats=N_repeats, fn_run=mab_bernoulli_greedy_run)
    regrets_a_lb = mab_bernoulli_low_bound(mu_a, N_actions=N_actions, N_episodes=N_episodes)

    if plot_type == 'seaborn':
        regrets_d = {'TS':regrets_a_ts_repeat, 'Greedy':regrets_a_greedy_repeat}    
        sns_lineplot_dic(regrets_d)
    else:
        plt.semilogx(np.mean(regrets_a_ts_repeat,axis=0), label='TS')
        plt.semilogx(regrets_a_lb, label='LB')
        plt.xlabel('T')
        plt.ylabel('regrets')
        plt.legend(loc=0)
        plt.show()
    plt.show()


def main():
    # mab_bernoulli_ts(plot_type='matplotlib')
    # mab_bernoulli_fn(plot_type='matplotlib')
    mab_bernoulli_fn(plot_type='seaborn')


if __name__ == '__main__':
    main()
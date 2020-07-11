import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from locallib import input_default_with 


class RL_System:
    def __init__(self, N_S, N_A, lr_alpha = 0.1):
        self.Qsa = np.zeros((N_S, N_A), dtype=np.float) + 0.5
        self.Vs = np.zeros(N_S, dtype=np.float) + 0.5
        self.lr_alpha = lr_alpha

    def update_Qsa_mc(self, replay_buff_d):
        discounted_return = np.array(replay_buff_d['reward'])
        calc_discounted_return_inplace(discounted_return)

        for idx in range(len(discounted_return)):
            S = replay_buff_d['S'][idx]
            action = replay_buff_d['action'][idx]
            mc_error = discounted_return[idx] - self.Qsa[S, action]
            self.Qsa[S, action] += self.lr_alpha * mc_error

    def update_Vs_mc(self, replay_buff_d):
        discounted_return = np.array(replay_buff_d['reward'])
        calc_discounted_return_inplace(discounted_return)

        for idx in range(len(discounted_return)):
            S = replay_buff_d['S'][idx]
            # action = replay_buff.d['action'][idx]
            mc_error = discounted_return[idx] - self.Vs[S]
            self.Vs[S] += self.lr_alpha * mc_error

    def update_mc(self, replay_buff_d):
        self.update_Qsa_mc(replay_buff_d)
        self.update_Vs_mc(replay_buff_d)

    def update_Qsa_td(self, replay_buff_d, gamma = 1.0):
        S = replay_buff_d['S'][0]
        action = replay_buff_d['action'][0]
        reward = replay_buff_d['reward'][0]
        done = replay_buff_d['done'][0]
        for idx in range(1, len(replay_buff_d['S'])):
            # SARSA
            S_new = replay_buff_d['S'][idx]
            action_new = replay_buff_d['action'][idx]
            td_error = (reward + (1-done) * gamma * self.Qsa[S_new, action_new]) - self.Qsa[S, action]
            self.Qsa[S, action] += self.lr_alpha * td_error        
            S = S_new
            action = action_new
            reward = replay_buff_d['reward'][idx]
            done = replay_buff_d['done'][idx]
        td_error = reward - self.Qsa[S, action]
        self.Qsa[S, action] += self.lr_alpha * td_error

    def update_Vs_td(self, replay_buff_d, gamma = 1.0):
        for idx in range(len(replay_buff_d['S'])):
            S = replay_buff_d['S'][idx]
            # action = replay_buff_d['S'][idx]
            S_new = replay_buff_d['S_new'][idx]
            reward = replay_buff_d['reward'][idx]
            done = replay_buff_d['done'][idx]
            td_error = (reward + (1-done) * gamma * self.Vs[S_new]) - self.Vs[S]
            self.Vs[S] += self.lr_alpha * td_error        

    def update_td(self, replay_buff_d):
        self.update_Qsa_td(replay_buff_d)
        self.update_Vs_td(replay_buff_d)


class RandomWalkEnv:
    def __init__(self):
        self.N_S = 5 # 0, 1, 2, 3, 4 (A,B,C,D,E)
        self.S = 2 
        self.action_value_list = np.array([-1, 1]) # left: -1, right: +1
        self.N_A = len(self.action_value_list)

        self.reward_table = np.zeros((self.N_S, self.N_A), dtype=np.float)
        self.reward_table[4, 1] = 1.0

        self.done_table = np.zeros((self.N_S, self.N_A), dtype=np.int)
        self.done_table[4, 1] = 1
        self.done_table[0, 0] = 1

    def reset_internal(self):
        self.S = 2

    def reset(self):
        self.reset_internal()
        return self.S

    def sample_action(self):
        action = np.random.randint(self.N_A)
        return action

    def step(self, action):
        reward = self.reward_table[self.S, action]
        done = self.done_table[self.S, action]
        if done:
            self.reset_internal()
        else:    
            self.S += self.action_value_list[action]
        return self.S, reward, done


@jit
def calc_discounted_return_inplace(discounted_return, gamma=1.0):
    r_future = 0.0
    for idx in range(len(discounted_return)):
        discounted_return[-idx] += gamma * r_future
        r_future = discounted_return[-idx]


class ReplayBuff:
    def __init__(self):
        self.reset()
    
    def append(self, S, action, S_new, reward, done):
        """
        Separated copies are needed to save, so that we use copy() command
        """
        self.d['S'].append(S)
        self.d['action'].append(action)
        self.d['S_new'].append(S_new)
        self.d['reward'].append(reward)
        self.d['done'].append(1 if done else 0)                

    def reset(self):
        self.d = {'S':[], 'action': [], 'S_new': [], 'reward': [], 'done': []}


class ReplayBuff_PG:
    def __init__(self):
        self.reset()
    
    def append(self, S, action, S_new, reward, done, prob):
        """
        Separated copies are needed to save, so that we use copy() command
        """
        self.d['S'].append(S)
        self.d['action'].append(action)
        self.d['S_new'].append(S_new)
        self.d['reward'].append(reward)
        self.d['done'].append(1 if done else 0)          
        self.d['prob'].append(prob)      

    def reset(self):
        self.d = {'S':[], 'action':[], 'S_new':[], 'reward':[], 'done':[], 'prob':[]}


class RL_System_PG(RL_System):
    def __init__(self, N_S, N_A, lr_alpha = 0.1):
        super(RL_System_PG, self).__init__(N_S, N_A, lr_alpha=lr_alpha)

        assert N_A == 2, 'N_A=2 is supported only.'
        self.function_approx_binary = np.zeros(N_S, dtype=np.float) + 0.5 

    def get_action_prob(self, S):        
        prob_action_0 = self.function_approx_binary[S]
        sample = np.random.uniform()
        if sample < prob_action_0:
            return 0, prob_action_0 
        else:
            return 1, 1 - prob_action_0

    def update_Qsa_mc(self, replay_buff_d):
        discounted_return = np.array(replay_buff_d['reward'])
        calc_discounted_return_inplace(discounted_return)

        for idx in range(len(discounted_return)):
            S = replay_buff_d['S'][idx]
            action = replay_buff_d['action'][idx]
            mc_error = discounted_return[idx] - self.Qsa[S, action]
            self.Qsa[S, action] += self.lr_alpha * mc_error

    def update_pg(self, replay_buff_d):
        discounted_return = np.array(replay_buff_d['reward'])
        calc_discounted_return_inplace(discounted_return)
        prob = np.array(replay_buff_d['prob'], dtype=np.float)
        
        # Total performance regardless of 
        # performance = np.log(prob) * discounted_return

    def update_mc(self, replay_buff_d):
        self.update_pg(replay_buff_d)

        self.update_Qsa_mc(replay_buff_d)
        self.update_Vs_mc(replay_buff_d)


class DNN_Random_Walk(tf.keras.layers.Layer):
    def __init__(self, N_A=2):
        super(DNN_Random_Walk, self).__init__()
        self.linear = tf.keras.layers.Dense(N_A)

    def call(self, x):
        logits = self.linear(x)
        probs = tf.nn.softmax(logits)
        return probs

class RL_System_PG_TF(RL_System_PG):
    def __init__(self, N_S, N_A, lr_alpha = 0.1):
        super(RL_System_PG_TF, self).__init__(N_S, N_A, lr_alpha=lr_alpha)
        
        self.N_S = N_S
        self.N_A = N_A
        self.function_approx = DNN_Random_Walk(N_A=N_A)

    def get_action_prob_tf(self, S):
        """
        Return
        ------
        action, probs_tf
        """        
        S_a = np.zeros((1,self.N_S), dtype=np.float16)
        S_a[0, S] = 1.0
        S_tf = tf.Variable(S_a)
        probs_tf = self.function_approx(S_tf)
        action = np.random.choice(self.N_A, p=probs_tf.numpy()[0])
        return action, tf.reshape(probs_tf[:,action], (-1,1))

    def get_policy(self, S):
        S_a = np.zeros((1,self.N_S), dtype=np.float16)
        S_a[0, S] = 1.0
        S_tf = tf.Variable(S_a)
        probs_tf = self.function_approx(S_tf)
        return probs_tf

    def update_Qsa_mc(self, replay_buff_d):
        discounted_return = np.array(replay_buff_d['reward'])
        calc_discounted_return_inplace(discounted_return)

        for idx in range(len(discounted_return)):
            S = replay_buff_d['S'][idx]
            action = replay_buff_d['action'][idx]
            mc_error = discounted_return[idx] - self.Qsa[S, action]
            self.Qsa[S, action] += self.lr_alpha * mc_error

    def learning_pg(self, replay_buff_d, tape, optimizer):
        discounted_return = np.array(replay_buff_d['reward'])
        calc_discounted_return_inplace(discounted_return)
        prob = tf.concat(replay_buff_d['prob'], 0)
        performance_vec = tf.math.log(prob) * discounted_return.reshape(-1,1)
        performance = tf.reduce_sum(performance_vec)
        gradients = tape.gradient(-performance, self.function_approx.trainable_weights)
        optimizer.apply_gradients(zip(gradients, self.function_approx.trainable_weights))

    def update_mc(self, replay_buff_d):
        self.update_Qsa_mc(replay_buff_d)
        self.update_Vs_mc(replay_buff_d)


def _randomwalk_run_pg_learning(N_episodes=1, learning_mode=True, lr=0.01):
    print('Plocy gradient mode (Under development)')
    N_S = 5
    N_A = 2
    rl_system = RL_System_PG_TF(N_S, N_A)
    random_walk_env = RandomWalkEnv()
    S = random_walk_env.reset()

    optimizer = tf.keras.optimizers.SGD()
    replay_buff = ReplayBuff_PG()
    for _ in range(N_episodes):
        with tf.GradientTape() as tape:
            done = False        
            while not done:
                action, prob = rl_system.get_action_prob_tf(S)
                S_new, reward, done = random_walk_env.step(action)
                replay_buff.append(S, action, S_new, reward, done, prob)
                S = S_new
            
            rl_system.update_mc(replay_buff.d)
            if learning_mode:
                rl_system.learning_pg(replay_buff.d, tape, optimizer)
        replay_buff.reset()
    
    print('Qsa:')
    print(rl_system.Qsa)
    print('Vs:')
    print(rl_system.Vs)

    print('Policy:')
    for S in range(N_S):
        print(f'Policy({S}) =', rl_system.get_policy(S).numpy())


def randomwalk_run_pg_learning(N_episodes=2000, learning_mode=True, lr = 0.01, disp_flag = True):
    print('Plocy gradient mode (Under development)')
    N_S = 5
    N_A = 2
    rl_system = RL_System_PG_TF(N_S, N_A)
    random_walk_env = RandomWalkEnv()
    S = random_walk_env.reset()

    #optimizer = tf.keras.optimizers.SGD(lr=lr)
    optimizer = tf.keras.optimizers.Adam(lr=lr)
    replay_buff = ReplayBuff_PG()
    policy_array= np.zeros((N_episodes+1,N_S, N_A),dtype=float)
    for episode in range(N_episodes):
        if disp_flag:
            #print('Before episode ', episode)
            for S in range(N_S):
                #print(f'Policy({S}) =', rl_system.get_policy(S).numpy())    
                policy_array[episode, S] = rl_system.get_policy(S).numpy()

        with tf.GradientTape() as tape:
            done = False        
            while not done:
                action, prob = rl_system.get_action_prob_tf(S)
                S_new, reward, done = random_walk_env.step(action)
                replay_buff.append(S, action, S_new, reward, done, prob)
                S = S_new

            rl_system.update_mc(replay_buff.d)
            if learning_mode:
                rl_system.learning_pg(replay_buff.d, tape, optimizer)
        replay_buff.reset()

    print('Qsa:')
    print(rl_system.Qsa)
    print('Vs:')
    print(rl_system.Vs)

    print('After learning:')
    for S in range(N_S):
        print(f'Policy({S}) =', rl_system.get_policy(S).numpy())
        policy_array[N_episodes, S] = rl_system.get_policy(S).numpy()

    if disp_flag:
        plt.figure(figsize=(15,7))
        plt.subplot(1,2,1)
        plt.title('Policy array for (State,action=0)')
        plt.plot(policy_array[:,0,0], label="0,0")
        plt.plot(policy_array[:,1,0], label="1,0")
        plt.plot(policy_array[:,2,0], label="2,0")
        plt.plot(policy_array[:,3,0], label="3,0")
        plt.plot(policy_array[:,4,0], label="4,0")
        plt.legend(loc=0)
        plt.subplot(1,2,2)
        plt.title('Policy array for (State,action=1)')
        plt.plot(policy_array[:,0,1], label="0,1")
        plt.plot(policy_array[:,1,1], label="1,1")
        plt.plot(policy_array[:,2,1], label="2,1")
        plt.plot(policy_array[:,3,1], label="3,1")
        plt.plot(policy_array[:,4,1], label="4,1")
        plt.legend(loc=0)
        plt.show()


def randomwalk_run(N_episodes=1, rl_mode='td'):
    rl_system = RL_System(5, 2)
    random_walk_env = RandomWalkEnv()
    S = random_walk_env.reset()
    # print(f'Current state: {S}')

    replay_buff = ReplayBuff()
    for _ in range(N_episodes):
        done = False
        while not done:
            action = random_walk_env.sample_action()
            S_new, reward, done = random_walk_env.step(action)
            replay_buff.append(S, action, S_new, reward, done)
            # print(f'S:{S}, action:{action}, S_new:{S_new}, reward:{reward}, done:{done}')
            S = S_new

        # print("Replay buff:")
        # print(replay_buff.d)

        if rl_mode == 'mc':
            rl_system.update_mc(replay_buff.d)
        else: # rl_mode == 'td' (default mode)
            rl_system.update_td(replay_buff.d)
            
        replay_buff.reset()
    print('Qsa:')
    print(rl_system.Qsa)
    print('Vs:')
    print(rl_system.Vs)


def main():
    print('Testing random walk')
    N_episodes = input_default_with('How many episodes do you want to run?', 10)
    rl_mode_index = input_default_with('Which mode do you want to use?(0=td,1=mc,2=pg_learning)', 0)
    rl_mode = ['td', 'mc', 'pg'][rl_mode_index]
    if rl_mode == 'pg':
        randomwalk_run_pg_learning(N_episodes=N_episodes, disp_flag = True)
    else:
        randomwalk_run(N_episodes=N_episodes, rl_mode=rl_mode)


if __name__ == "__main__":
    # This is the main function.
    main()

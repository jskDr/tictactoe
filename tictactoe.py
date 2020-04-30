import tensorflow as tf
# from tensorflow.keras.layers import Layer, Dense
import numpy as np
import matplotlib.pyplot as plt
import pickle
import numba
from numba import jit
import random
from typing import List, Tuple, Union

# TicTacToe game has nine stateus with nine actions. 
# An user can put his ston on any postion in the borad except 

class EnvModel_TBL: # table looup model (basic model)   
    def __init__(self, N_A, N_Symbols):
        """
        tbl: table look-up
        """
        self.N_A = N_A
        self.N_Symbols = N_Symbols        
        self.S_idx_base = np.power(self.N_Symbols,range(self.N_A))
        All_S = np.power(self.N_Symbols,self.N_A)

        # Model: All states x all actions x all associated new states x (reward, prob)
        self.N_Model = np.zeros((All_S, self.N_A), dtype=np.int) #(Reward, p)
        self.Model_R_N = np.zeros((All_S, self.N_A), dtype=np.float32) #(Reward, p)
        self.Model_Done1_N = np.zeros((All_S, self.N_A), dtype=np.float32) #(Reward, p)
        self.Model_P_N = np.zeros((All_S, self.N_A, self.N_A), dtype=np.float32) #(Reward, p)
        self.Model_Done2_N = np.zeros((All_S, self.N_A, self.N_A), dtype=np.float32) #(Reward, p)

    def update(self, S, action, reward, op_action, done):
        S_idx = np.sum(self.S_idx_base * S)
        self.Model_R_N[S_idx, action] += reward
        self.Model_Done1_N[S_idx, action] += 1 if done else 0
        self.N_Model[S_idx, action] += 1
        if op_action is not None:
            # Model_Done1_N is 1, then Done2_N is no need to consider. 
            self.Model_P_N[S_idx, action, op_action] += 1.0                    
            self.Model_Done2_N[S_idx, action, op_action] += 1 if done else 0

    def sampling(self, shuffle_Replay_buff_d, P_no):
        # generated_Replay_buff_d = {'S':[], 'action':[], 'reward':[], 'S_new':[]}
        generated_Replay_buff = ReplayBuff()
        for j in range(len(shuffle_Replay_buff_d['S'])):
            S = shuffle_Replay_buff_d['S'][j]
            action = shuffle_Replay_buff_d['action'][j]
            S_idx = np.sum(self.S_idx_base * S)
            reward = self.get_Model_R(S_idx, action)
            S_new, done = self.get_Model_S_new(S, action, P_no, S_idx)
            generated_Replay_buff.append(S.copy(), action, S_new.copy(), reward, done)
        return generated_Replay_buff.d

    def check(self):
        print()
        print('Checking modeling results')    
        Done = False
        while not Done:
            S_str = input_default('Enter State vector (S)?(defulat=[0,0,0,0,0,0,0,0,0]) ', '[0,0,0,0,0,0,0,0,0]', str)
            S = eval(S_str)
            action = input_default('Enter action?(defulat=0) ', 0, int)
            self.disp(S, action)
            yn = input_default('Quit=0 or test again=1?(defalut=0) ', 0, int)
            Done = not yn

    def disp(self, S, action=0):
        S_idx = np.sum(self.S_idx_base * S)
        print(f'S = {S} --> S_idx = {S_idx}')
        print(f'N_Model[{S},:] = ', self.N_Model[S_idx,:])
        print(f'Model_R[{S},:] = ', [self.get_Model_R(S_idx,i) for i in range(9)])
        print(f'Model_P[{S},{action},:] = ', [self.get_Model_P(S_idx,action,i) for i in range(9)])

    def get_Model_P(self, S_idx, action, op_action):
        """get_Model_P(self, S_idx, action, op_action):
        """
        if self.N_Model[S_idx, action]:
            return self.Model_P_N[S_idx, action, op_action] / self.N_Model[S_idx, action]
        else:
            return 0.0

    def get_Model_S_new(self, S, action, P_no, S_idx):
        S_half = S.copy()
        S_half[action] = P_no
        op_action_list = find_remained_action_list(S_half)
        done = 1 if len(op_action_list) < 2 else 0 # if 1, op_action will be append

        if len(op_action_list) > 0:
            P = []
            for op_action in op_action_list:
                P.append(self.get_Model_P(S_idx, action, op_action))
            op_action = op_action_list[np.argmax(np.random.multinomial(1, P))]
            P_no_op = 3 - P_no
            S_half[op_action] = P_no_op
        return S_half, done

    def get_Model_R(self, S_idx, action):
        if self.N_Model[S_idx, action]:
            return self.Model_R_N[S_idx, action] / self.N_Model[S_idx, action]
        else:
            return 0.0

    def get_Model_Done1(self, S_idx, action):
        if self.N_Model[S_idx, action]:
            return self.Model_Done1_N[S_idx, action] / self.N_Model[S_idx, action]
        else:
            return 0.0            

    def get_Model_Done2(self, S_idx, action, op_action):
        if self.Model_Done1_N[S_idx, action]:
            return self.Model_Done2_N[S_idx, action, op_action] / self.Model_Done1_N[S_idx, action] 
        else:
            return 0.0

class CntTrace:
    def __init__(self):
        self.buff = []
        self.cnt = [0] * 9
        self.buff.append(self.cnt.copy())

    def calc_and_append(self, reward, play_order):
        if reward == 1.0: 
            self.cnt[1] += 1               # P_no = 1 
            self.cnt[2 + play_order] += 1  # player_order = 1
            self.cnt[4 + play_order] += 1  # player_order | P_no = 1, so it occupied 2 lists
        elif reward == 0.5:
            self.cnt[0] += 1 
        else: # play_order = 2
            self.cnt[2] += 1                   # P_no = 2  
            self.cnt[2 + 3 - play_order] += 1  # player_order = 2
            self.cnt[6 + 3 - play_order] += 1  # player_order | P_no = 2, so it start from 6
        self.buff.append(self.cnt.copy())

    def disp(self):
        print('cnt: ', self.cnt)

class color:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'

def S_to_action_list(S, N_A=9) -> Tuple[List[int], int]:
    """
    S to action_list
    """
    action_list = []
    no_occupied = 0
    for a in range(N_A):
        if S[a] != 0:
            action_list.append(a)
            no_occupied += 1
    return action_list, no_occupied

def find_remained_action_list(S, N_A=9) -> Tuple[List[int], int]:
    """
    S to action_list
    """
    action_list = []
    for a in range(N_A):
        if S[a] == 0:
            action_list.append(a)
    return action_list 

def random_shuffle_dict_inplace(Replay_buff):
    """Shuffle dict with same odering
    """
    key0 = list(Replay_buff.keys())[0]
    Replay_buff_idx_list = list(range(len(Replay_buff[key0])))
    random.shuffle(Replay_buff_idx_list) # inplace shuffling
    # print(Replay_buff_idx_list)
    for key in Replay_buff:
        Replay_buff[key] = [Replay_buff[key][i] for i in Replay_buff_idx_list]

@jit
def calc_S_idx_numba(S:np.ndarray, N_Symbols:int=3) -> int:
    S_idx = 0
    unit = 1
    for i in range(len(S)):
        S_idx += unit*S[i]
        unit *= N_Symbols
    return S_idx


def set_state_inplace(S, action, P_no): 
    """ 
    set_state_inplace(S, action, P_no)

    [Inputs] 
    S is numpy array.
    """
    # assert S[action] == 0, 'position should be empty to put a new stone' 
    S[action] = P_no # User numpy to insert action in the specific position


def set_state(S, action, P_no): 
    """ 
    set_state(S, action, P_no)

    [Inputs] 
    S is numpy array.

    [Returns]
    Sa is S with new action.
    """
    # assert S[action] == 0, 'position should be empty to put a new stone' 
    Sa = S.copy()
    Sa[action] = P_no # User numpy to insert action in the specific position
    return Sa


@jit # No speed advantage compared to pure Python code
def set_state_inplace_numba(S, action, P_no): 
    """ 
    set_state_inplace_numba(S, action, P_no)
    - No speed advantage compared to pure Python code

    [Inputs] 
    S is numpy array.
    """
    # assert S[action] == 0, 'position should be empty to put a new stone' 
    S[action] = P_no # User numpy to insert action in the specific position
    

def calc_reward(S):
    mask_l = tf.constant([[1,1,1,0,0,0,0,0,0], [0,0,0,1,1,1,0,0,0], [0,0,0,0,0,0,1,1,1],
                         [1,0,0,1,0,0,1,0,0], [0,1,0,0,1,0,0,1,0], [0,0,1,0,0,1,0,0,1],
                         [1,0,0,0,1,0,0,0,1], [0,0,1,0,1,0,1,0,0]], dtype=tf.int16)
    for mask in mask_l:
        # print(mask)
        mask_S = mask * S
        # print(mask_S)

        for player in [1,2]:
            abs_err = tf.reduce_sum(tf.abs(mask_S - player * mask))
            # print(abs_err)
            if abs_err == 0:
                # print(f'Player{player} wins')
                return player
    return 0    


def calc_reward_tf(S):
    mask_l = tf.constant([[1,1,1,0,0,0,0,0,0], [0,0,0,1,1,1,0,0,0], [0,0,0,0,0,0,1,1,1],
                         [1,0,0,1,0,0,1,0,0], [0,1,0,0,1,0,0,1,0], [0,0,1,0,0,1,0,0,1],
                         [1,0,0,0,1,0,0,0,1], [0,0,1,0,1,0,1,0,0]], dtype=tf.int32)

    S = tf.constant(S, dtype=tf.int32)
    S = tf.reshape(S, shape=(1,-1))
    S_cp = tf.matmul(tf.ones((mask_l.shape[0],1),dtype=tf.int32), S)
    mask_S = mask_l * S_cp 

    for player in [1, 2]:
        if tf.reduce_any(tf.reduce_sum(tf.abs(mask_S - player * mask_l),axis=1) == 0):
            return player
    
    return 0

MASK_L = np.array([[1,1,1,0,0,0,0,0,0], [0,0,0,1,1,1,0,0,0], [0,0,0,0,0,0,1,1,1],
                         [1,0,0,1,0,0,1,0,0], [0,1,0,0,1,0,0,1,0], [0,0,1,0,0,1,0,0,1],
                         [1,0,0,0,1,0,0,0,1], [0,0,1,0,1,0,1,0,0]], dtype=int)

@jit
def calc_reward_numba(mask_l, S):
    for i in range(len(mask_l)):
        mask = mask_l[i]
        mask_S = mask * S

        for player in [1,2]:
            abserr = np.sum(np.abs(mask_S - player * mask))
            if abserr == 0:
                return player
    return 0    


def _calc_reward_numba(mask_l, S):
    for i in range(len(mask_l)):
        mask = mask_l[i]
        mask_S = mask * S

        for player in [1,2]:
            abserr = np.sum(np.abs(mask_S - player * mask))
            if abserr == 0:
                return player
    return 0    


def one_of_amax(arr, disp_flag=False):
    results = np.where(arr == np.amax(arr))[0]

    if disp_flag:
        print('Equally max actions:', results)
    
    action = results[np.random.randint(0, len(results), 1)[0]]   
    return action


def _buff_depart(Buff, disp_flag=False):
    Buff_dual = [{'S':[], 'a':[], 'r':[], 'S_next':[]}, {'S':[], 'a':[], 'r':[], 'S_next':[]}]
    for i, (p, S, a, r, S_next) in enumerate(zip(Buff['P_no'], Buff['S'], Buff['a'], Buff['r'], Buff['S_next'])):
        if i > 0:
            # final reward for a player is reward of a next player
            prev_p = 2 if p==1 else 1
            #Buff_dual[prev_p-1]['r'][-1] = -r # 1 for player#2 --> -1 for player#1, vice versa
            Ratio_lose_per_win = 10.0
            Buff_dual[prev_p-1]['r'][-1] = -r * Ratio_lose_per_win
            if disp_flag:
                print('i, prev_p, Buff_dual[prev_p-1]')
                print(i, prev_p, Buff_dual[prev_p-1])
        Buff_dual[p-1]['S'].append(S)
        Buff_dual[p-1]['a'].append(a)
        Buff_dual[p-1]['r'].append(r)  
        Buff_dual[p-1]['S_next'].append(S_next)
        if disp_flag:
            print('i, p, Buff_dual[p-1]')
            print(i, p, Buff_dual[p-1])                
    return Buff_dual                


def buff_depart(Buff, disp_flag=False):
    """
    r: win -> 1, tie -> 0.5, lose -> 0
    Tie score (0,5) is spreaded to both players. 
    """
    Buff_dual = [{'S':[], 'a':[], 'r':[], 'S_next':[]}, {'S':[], 'a':[], 'r':[], 'S_next':[]}]
    for i, (p, S, a, r, S_next) in enumerate(zip(Buff['P_no'], Buff['S'], Buff['a'], Buff['r'], Buff['S_next'])):
        if i > 0:
            # final reward for a player is reward of a next player
            prev_p = 2 if p==1 else 1
            #Buff_dual[prev_p-1]['r'][-1] = -r # 1 for player#2 --> -1 for player#1, vice versa
            # Ratio_lose_per_win = 10.0 --> 1 since win=1, tie=0.5 (or 0), lose=0 (or -1)
            
            # means lost (win by opponet), actually no need to change r since it is already 0
            # if r == 1: # means lost (win by opponent)
            #    Buff_dual[prev_p-1]['r'][-1] = 0 
            if r == 0.5: #But if it is tie, both should have 0.5                
                Buff_dual[prev_p-1]['r'][-1] = 0.5 
            if disp_flag:
                print('i, prev_p, Buff_dual[prev_p-1]')
                print(i, prev_p, Buff_dual[prev_p-1])
        Buff_dual[p-1]['S'].append(S)
        Buff_dual[p-1]['a'].append(a)
        Buff_dual[p-1]['r'].append(r)  
        Buff_dual[p-1]['S_next'].append(S_next)
        if disp_flag:
            print('i, p, Buff_dual[p-1]')
            print(i, p, Buff_dual[p-1])                
    return Buff_dual                


def discounted_inplace(Buff_r, ff):
    """discounted_inplace(Buff_r): 
    Convert a reward vector to a discounted return vector using ff,
    where ff means forgeting factor.

    [Input] Buff_r = Buff[r]: stores rewards in each episode
    """
    g_prev = 0
    for i, r_l in enumerate(reversed(Buff_r)):
        # print(g_prev, i, r_l, Buff_r)
        Buff_r[-i-1] = r_l + ff * g_prev
        g_prev = Buff_r[-i-1]


class Q_System:
    def __init__(self, N_A=9, N_Symbols=3, epsilon=0.01, disp_flag=False):
        """
        Q_System(N_A=9, N_Symbols=3, epsilon=0.01, disp_flag=False
        
        Input
        -----
        N_A : Number of actions

        N_Symbols : Number of possible symbols in each point: 0, 1, 2,
        representing empty, player1, player2

        N_S : Number of states
        """        
        if N_A is not None:
            self.disp_flag = disp_flag
            N_S = N_Symbols**N_A
            self.Qsa = [np.zeros((N_S,N_A)), np.zeros((N_S,N_A))]       
            self.N_A = N_A
            self.N_Symbols = N_Symbols
            self.epsilon = epsilon
        else:
            self.disp_flag = False
            self.Qsa = None      
            self.N_A = None
            self.N_Symbols = None
            self.epsilon = None

    def save(self):
        f = open('tictactoe_data.pckl', 'wb')
        obj = [self.N_A, self.N_Symbols, self.epsilon, self.Qsa, 'Q_System']
        pickle.dump(obj, f)
        f.close()

    def load(self):
        f = open('tictactoe_data.pckl', 'rb')
        obj = pickle.load(f)
        [self.N_A, self.N_Symbols, self.epsilon, self.Qsa, self.class_name] = obj
        f.close()

    def print_s_a(self, S, action_list):
        Sq_N_A = int(np.sqrt(self.N_A))
        Sa = []
        for s in S:
            if s == 1:
                Sa.append('A')
            elif s == 2:
                Sa.append('B')
            else:
                Sa.append('')
        for a in action_list:
            Sa[a] = f'{a}'
        Sa_array = np.array(Sa).reshape(-1,Sq_N_A)
        print()
        print('-------------------')
        for ii in range(Sq_N_A):
            for jj in range(Sq_N_A):
                if Sa_array[ii,jj] == 'A':
                    print(color.BLUE + color.BOLD + f'{Sa_array[ii,jj]}' + color.END, end=' ')
                elif Sa_array[ii,jj] == 'B':
                    print(color.RED + color.BOLD + f'{Sa_array[ii,jj]}' + color.END, end=' ')
                else:
                    print(f'{Sa_array[ii,jj]}', end=' ')
            print()
        print('-------------------')
        print()

    def check_available_win_action(self, P_no, S, action_list):
        """
        yes, action = check_avaiable_win_action(S, action_list)
        - check whether there is any action which can make a game win right away.
        All avaiable action will be checked up. 
        """ 
        for action in action_list:
            Sa = S.copy()
            Sa[action] = P_no 
            reward = calc_reward_numba(MASK_L, Sa)
            # print('Checking available win:P_no, S, Sa, action', P_no, S, Sa, action)
            if reward:
                # print('Founded available win: Sa, action-->', Sa, action)
                return True, action
        return False, None

    def check_lose_protection_action(self, P_no, S, action_list):
        """
        yes, action = check_lose_protection_action(P_no, S, action_list)
        - check whether there is need for the imidiate action to protect lose
        by the proponent's next move.

        - calc_reward_tf --> calc_reward_numba: to upgrade the computation speed.
        """ 
        for action in action_list:
            Sa = S.copy()
            player_op = 1 if P_no==2 else 2
            Sa[action] = player_op
            reward = calc_reward_numba(MASK_L, Sa)
            if reward:
                # print('Founded lose protection: Sa, action -->', Sa, action)
                return True, action
        return False, None

    def calc_S_idx(self, S):
        S_idx = 0
        unit = 1
        for s in S:
            S_idx += s*unit
            unit *= self.N_Symbols
        return S_idx
        
    def _policy_random(self, S, a):
        return 1 / self.N_A

    def policy_random(self, P_no, S, action_list): 
        action_prob = []
        S_idx = self.calc_S_idx(S)
        for _ in action_list:
            action_prob.append(1/len(action_list))        
        action_idx = tf.squeeze(tf.random.categorical(tf.math.log([action_prob]),1)).numpy()
        if action_idx == len(action_prob): # if all zeros in action_prob
            action = action_list[tf.squeeze(np.random.randint(0, len(action_list), 1))]
        else:
            action = action_list[action_idx]
        if self.disp_flag:
            print('S_idx', S_idx, 'action', action, 'action_list', action_list, 'action_prob', action_prob)
        return action
    
    def policy(self, P_no:int, S:List[int], action_list:List[int]):
        """Return action regardless P_no but just specify Q[P_no]
        """
        action_prob = []
        S_idx = self.calc_S_idx(S)
        for a in action_list:
            action_prob.append(self.Qsa[P_no-1][S_idx,a])                

        # We consider max Q with epsilon greedy
        if tf.squeeze(tf.random.uniform([1,1])) > self.epsilon:
            action = action_list[one_of_amax(action_prob)]            
        else:
            action = action_list[np.random.randint(0,len(action_list),1)[0]]
            
        if self.disp_flag:
            print('S_idx', S_idx, 'action', action, 
                  'action_list', action_list, 'action_prob', action_prob)
        return action

    def _r0_policy(self, P_no, S, action_list):
        action_prob = []
        S_idx = self.calc_S_idx(S)
        for a in action_list:
            action_prob.append(self.Qsa[P_no-1][S_idx, a])        
        action_idx = tf.squeeze(tf.random.categorical(tf.math.log([action_prob]),1)).numpy()
        if action_idx == len(action_prob): # if all zeros in actoin_prob
            action = action_list[tf.squeeze(np.random.randint(0, len(action_list), 1))]
        else:
            action = action_list[action_idx]
        if self.disp_flag:
            print('S_idx', S_idx, 'action', action, 'action_list', action_list, 'action_prob', action_prob)

        return action

    def find_action_list(self, S) -> Tuple[List[int], int]:
        """
        S to action_list
        """
        action_list = []
        no_occupied = 0
        for a in range(self.N_A):
            if S[a] == 0:
                action_list.append(a)
            else:
                no_occupied += 1
        return action_list, no_occupied
    
    # Take action_prob at the given state
    def get_action(self, P_no, S):
        """
        Return action, done
        """
        action_list, no_occupied = self.find_action_list(S)
        # Since number of possible actions are reduced, 
        # denominator is also updated. 
        action = self.policy(P_no, S, action_list)
        done = no_occupied == (self.N_A - 1)
        return action, done

    def get_action_with_random(self, P_no, S):
        """Return action, done
        """
        action_list, no_occupied = self.find_action_list(S)
        # Since number of possible actions are reduced, 
        # denominator is also updated. 
        if P_no == 1:
            action = self.policy(P_no, S, action_list)
        else:
            action = self.policy_random(P_no, S, action_list)
        done = no_occupied == (self.N_A - 1)
        return action, done

    def get_action_against_human(self, P_no, S):
        """Return action, done
        """
        action_list, no_occupied = self.find_action_list(S)
        # Since number of possible actions are reduced, 
        # denominator is also updated. 
        yes, action = self.check_available_win_action(P_no, S, action_list)
        if yes == False:
            yes, action = self.check_lose_protection_action(P_no, S, action_list)
            if yes == False:
                action = self.policy(P_no, S, action_list)
        if self.disp_flag: 
            self.show_Qsa(action_list, P_no, S)                    
        # print('[Agent-Qsa]', [f'{self.Qsa[P_no-1][S_idx,a]:.1e}' for a in action_list])
        # print('Agent action:', action)
        done = no_occupied == (self.N_A - 1)
        return action, done

    def show_Qsa(self, action_list, P_no, S):
        if self.Qsa: 
            S_idx = self.calc_S_idx(S)
            print('[Qsa]', [f'{a}:{self.Qsa[P_no-1][S_idx,a]:.1e}' for a in action_list])

    def get_action_with_human(self, P_no, S):
        """
        action, done = get_action_with_human(self, P_no, S)
        - Playing with human
        
        [Inputs]

        P_no : Human player index which represents 1=fist, 2=second playing
        """
        action_list, no_occupied = self.find_action_list(S)
        # Since number of possible actions are reduced, 
        # denominator is also updated. 
        print('The current game state is:')
        self.print_s_a(S, action_list)
        self.show_Qsa(action_list, P_no, S)

        rand_idx = np.random.randint(0, len(action_list))
        random_action = action_list[int(rand_idx)]

        action = None
        while action not in action_list:
            action = input_default(f'Type your action (default={random_action}): ', random_action, int)
            if action not in action_list:
                print('Type action again among in the avaible action list:', action_list)

        done = no_occupied == (self.N_A - 1)
        return action, done

    def play_with_random(self, P_no):
        """ 
        Buff = play_with_random(self, P_no)
        - Learning by playing with a random policy agent.
        
        [Inputs]  
            P_no: player number, which is 1 or 2

        [Returns]
            Buff = {'P_no': [], 'S':[], 'a':[], 'r':[], 'S_next': []}: gathered information during learning
                where S, a, r, S_next are state, action, rewrd, and next state

        [Examples]
            1. Buff = self.play(1)
            2. Buff = self.play(2)
        """
        N_A = self.N_A
        Buff = {'P_no': [], 'S':[], 'a':[], 'r':[], 'S_next': []}

        S = np.zeros((N_A,),dtype='int16') # #state == #action
        
        if self.disp_flag:
            print('S:', S)
        
        done = False
        while done == False:
            action, done = self.get_action_with_random(P_no, S)
            Buff['P_no'].append(P_no)
            Buff['S'].append(S.copy())
            Buff['a'].append(action)
            set_state_inplace(S, action, P_no)    
            Buff['S_next'].append(S.copy())   
            
            if self.disp_flag:
                print('S:', S)
            
            win_player = calc_reward_numba(MASK_L, S)
            reward = 0 if win_player == 0 else 1
            Buff['r'].append(reward)
            P_no = 1 if P_no == 2 else 2

            if win_player:
                done = True                

        if self.disp_flag:        
            if win_player:
                print(f'player {win_player} win')
            else:
                print(f'Tie game')

        return Buff    
   
    def play(self, P_no):
        """ 
        Buff = play(self, P_no)
        
        [Inputs]  
            P_no: player number, which is 1 or 2

        [Returns]
            Buff = {'P_no': [], 'S':[], 'a':[], 'r':[], 'S_next': []}: gathered information during learning
                where S, a, r, S_next are state, action, rewrd, and next state

        [Examples]
            1. Buff = self.play(1)
            2. Buff = self.play(2)
        """
        N_A = self.N_A
        Buff = {'P_no': [], 'S':[], 'a':[], 'r':[], 'S_next': []}

        S = np.zeros((N_A,),dtype='int16') # #state == #action
        
        if self.disp_flag:
            print('S:', S)
        
        done = False
        while done == False:
            action, done = self.get_action(P_no, S)

            Buff['P_no'].append(P_no)
            Buff['S'].append(S.copy())
            Buff['a'].append(action)
            set_state_inplace(S, action, P_no)    
            Buff['S_next'].append(S.copy())   
            
            if self.disp_flag:
                print('S:', S)
            
            win_player = calc_reward_numba(MASK_L, S)
            if win_player == 0:
                reward = 0.5 if done else 0 # if tie, reward -> 0.5, otherwise 0 yet 
            else: 
                reward = 1
            Buff['r'].append(reward)
            P_no = 1 if P_no == 2 else 2

            if win_player:
                done = True                

        if self.disp_flag:        
            if win_player:
                print(f'player {win_player} win')
            else:
                print(f'Tie game')

        return Buff    

    def play_by_scenario(self, P_no=1, action_list=[4, 1, 3, 5, 6, 2, 0]):
        """ 
        Buff = play(self, P_no=1, action_list=[4, 1, 3, 5, 6, 2, 0]):
        
        [Inputs]  
            P_no: player number, which is 1 or 2

        [Returns]
            Buff = {'P_no': [], 'S':[], 'a':[], 'r':[], 'S_next': []}: gathered information during learning
                where S, a, r, S_next are state, action, rewrd, and next state

        [Examples]
            1. Buff = self.play(1)
            2. Buff = self.play(2)
        """
        def get_action_by_scenario():
            action = action_list.pop(0)
            done = False if len(action_list) else True
            return action, done

        N_A = self.N_A
        Buff = {'P_no': [], 'S':[], 'a':[], 'r':[], 'S_next': []}

        S = np.zeros((N_A,),dtype='int16') # #state == #action
        
        if self.disp_flag:
            print('S:', S)
        
        done = False
        while done == False:
            action, done = get_action_by_scenario()
            Buff['P_no'].append(P_no)
            Buff['S'].append(S.copy())
            Buff['a'].append(action)
            set_state_inplace(S, action, P_no)    
            Buff['S_next'].append(S.copy())   
            
            if self.disp_flag:
                print('S:', S)
            
            win_player = calc_reward_numba(MASK_L, S)
            reward = 0 if win_player == 0 else 1
            Buff['r'].append(reward)
            P_no = 1 if P_no == 2 else 2

            if win_player:
                done = True                

        if self.disp_flag:        
            if win_player:
                print(f'player {win_player} win')
            else:
                print(f'Tie game')

        return Buff 


    def play_with_human(self, player_human=1):
        """ 
        Buff = play_with_human(self, P_no)
        - Playing with human
        
        [Inputs]  

        P_no: player number, which is 1 or 2

        [Returns]
        
        Buff = {'P_no': [], 'S':[], 'a':[], 'r':[], 'S_next': []}: gathered information during learning
        where S, a, r, S_next are state, action, rewrd, and next state

        [Examples]
            1. Buff = self.play(1)
            2. Buff = self.play(2)
        """
        N_A = self.N_A   
        Buff = {'P_no': [], 'S':[], 'a':[], 'r':[], 'S_next': []}

        S = np.zeros((N_A,),dtype='int16') # #state == #action
        
        if self.disp_flag:
            print('S:', S)

        P_no = 1 # set P_no = 1 while player_human could be 1 or 2        
        done = False
        while done == False:
            if player_human == P_no:
                action, done = self.get_action_with_human(P_no, S) 
            else:
                # P_no_trained_agent = 1 # random agent is 2
                # action, done = self.get_action_against_human(P_no_trained_agent, S)
                # Exact P_no is provided in order to check imediate win availablity
                action, done = self.get_action_against_human(P_no, S)
            Buff['P_no'].append(P_no)
            Buff['S'].append(S.copy())
            Buff['a'].append(action)
            set_state_inplace(S, action, P_no)    
            Buff['S_next'].append(S.copy())   
            
            if self.disp_flag:
                print('S:', S)
            
            win_player = calc_reward_numba(MASK_L, S)
            reward = 0 if win_player == 0 else 1
            Buff['r'].append(reward)
            P_no = 1 if P_no == 2 else 2

            if win_player:
                done = True                

        print(S.reshape(3,3))
        if win_player == player_human:
            print('You win')
        elif win_player != 0:
            print('You lose')
        else:
            print('Tie game')

        return Buff    


    def play_with_human_inference(self, player_human=1):
        """ 
        Buff = play_with_human(self, P_no)
        - Playing with human
        
        [Inputs]  

        P_no: player number, which is 1 or 2

        [Returns]
        
        Buff = {'P_no': [], 'S':[], 'a':[], 'r':[], 'S_next': []}: gathered information during learning
        where S, a, r, S_next are state, action, rewrd, and next state

        [Examples]
            1. Buff = self.play(1)
            2. Buff = self.play(2)
        """
        epsilon_now = self.epsilon
        self.epsilon = 0
        Buff = self.play_with_human(player_human=player_human)    
        self.epsilon = epsilon_now
        return Buff    


    def updateQsa_inplace(self, Qsa_player, Buff_player, lr):    
        if self.disp_flag:            
            print('---------------------------------------')
            print('S, S_idx, a, lr * r, Qsa_player[S_idx,a]')

        for S, a, r in zip(Buff_player['S'], Buff_player['a'], Buff_player['r']):        
            S_idx = self.calc_S_idx(S)
            Qsa_player[S_idx,a] += lr * r

            if self.disp_flag:
                print(S.reshape(-1, self.N_Symbols))
                print(S_idx, a, f'{lr * r:.1e}', f'{Qsa_player[S_idx,a]:.1e}')

    def update_Qsa_inplace(self, Buff, ff=0.9, lr=0.01):
        Buff_dual = buff_depart(Buff, disp_flag=self.disp_flag)
        
        # player#1
        for player in [1,2]:
            discounted_inplace(Buff_dual[player-1]['r'], ff) # for player#1

            if self.disp_flag:
                print('player:', player)
                print("Buff_dual[player-1]['r']", Buff_dual[player-1]['r'])
            
            self.updateQsa_inplace(self.Qsa[player-1], Buff_dual[player-1], lr)
            # updateQsa_stages_inplace(player, self.Qsa_stages[player-1], Buff_dual[player-1])

    def _learning(self, N_episodes=2, ff=0.9, lr=0.01, print_cnt=10):
        """Return: 
            cnt_trace = [cnt, ...]: cnt vector are stacked in cnt_trace
        """
        cnt = [0, 0, 0] # tie, p1, p2
        cnt_trace = [cnt.copy()]        
        player = 1
        for episode in range(N_episodes):
            # print('===================================')
            # Can save this data for play 2 as well
            
            # Decrease epsilon with respect to epside
            # self.epsilon = 1 - (1 + episode)/N_episodes
            self.epsilon = 1 / (1 + episode/100)
            # self.epsilon =  1 / (1 + episode)
            
            Buff = self.play(player)    
            self.update_Qsa_inplace(Buff, ff=ff, lr=lr)    
            win_player = 0 if Buff['r'][-1] == 0 else Buff['P_no'][-1]            
            cnt[win_player] += 1
            cnt_trace.append(cnt.copy())

            player = 2 if player == 1 else 1
            if episode % print_cnt == 0:
                print(episode, cnt)

                print('S = [0,0,0, 0,0,0, 0,0,0]')
                print('Qsa[0][0,:]', [f'{self.Qsa[0][0,a]:.1e}' for a in range(9)])
                print('Qsa[1][0,:]', [f'{self.Qsa[1][0,a]:.1e}' for a in range(9)])

                S = [1,1,0, 2,1,2, 1,2,2]
                S_idx = self.calc_S_idx(S)
                print('S = ', S)
                print(f'Qsa[0][{S_idx},:]', [f'{self.Qsa[0][S_idx,a]:.1e}' for a in range(9)])
                print(f'Qsa[1][{S_idx},:]', [f'{self.Qsa[1][S_idx,a]:.1e}' for a in range(9)])

                S = [1,1,0, 2,0,0, 2,0,0]
                S_idx = self.calc_S_idx(S)
                print('S = ', S)
                print(f'Qsa[0][{S_idx},:]', [f'{self.Qsa[0][S_idx,a]:.1e}' for a in range(9)])
                print(f'Qsa[1][{S_idx},:]', [f'{self.Qsa[1][S_idx,a]:.1e}' for a in range(9)])
        return cnt_trace

    def learning(self, N_episodes=2, ff=0.9, lr=0.01, print_cnt=10):
        """Return: 
            cnt_trace = [cnt, ...]: cnt vector are stacked in cnt_trace
        """
        cnt = [0, 0, 0] # tie, p1, p2
        cnt_trace = [cnt.copy()]        
        player = 1
        for episode in range(N_episodes):
            # print('===================================')
            # Can save this data for play 2 as well
            
            # Decrease epsilon with respect to epside
            self.epsilon = 1 / (1 + episode/N_episodes*9) 
            #self.epsilon = 1 / (1 + episode/N_episodes)
            # self.epsilon = 1 / (1 + episode/100)
            #self.epsilon =  1 / (1 + episode)
            
            Buff = self.play(player) # reward: any win --> 1, any tie --> 0.5, continue --> 0

            self.update_Qsa_inplace(Buff, ff=ff, lr=lr)    
            if Buff['r'][-1] == 0.5:
                cnt[0] += 1
            elif Buff['P_no'][-1] == player:
                cnt[1] += 1 # first playing player
            else:
                cnt[2] += 1 # second playing player

            cnt_trace.append(cnt.copy())

            player = 2 if player == 1 else 1
            if episode % print_cnt == 0:
                print(episode, cnt)                
                print('S = [0,0,0, 0,0,0, 0,0,0]')
                print('Qsa[0][0,:]', [f'{self.Qsa[0][0,a]:.1e}' for a in range(9)])
                print('Qsa[1][0,:]', [f'{self.Qsa[1][0,a]:.1e}' for a in range(9)])
                print('Exproration: Epsilon=', self.epsilon)
        return cnt_trace


    def _r1_learning(self, N_episodes=2, ff=0.9, lr=0.01, print_cnt=10):
        """Return: 
            cnt_trace = [cnt, ...]: cnt vector are stacked in cnt_trace
        """
        cnt = [0, 0, 0] # tie, p1, p2
        cnt_trace = [cnt.copy()]        
        player = 1
        for episode in range(N_episodes):
            # print('===================================')
            # Can save this data for play 2 as well
            
            # Decrease epsilon with respect to epside
            self.epsilon = 1 / (1 + episode/N_episodes)
            # self.epsilon = 1 / (1 + episode/100)
            #self.epsilon =  1 / (1 + episode)
            
            Buff = self.play(player)    
            self.update_Qsa_inplace(Buff, ff=ff, lr=lr)    
            win_player = 0 if Buff['r'][-1] == 0 else Buff['P_no'][-1]            
            cnt[win_player] += 1
            cnt_trace.append(cnt.copy())

            player = 2 if player == 1 else 1
            if episode % print_cnt == 0:
                print(episode, cnt)                
                print('S = [0,0,0, 0,0,0, 0,0,0]')

                norm_Q0_list = [self.Qsa[0][0,a] / np.max([self.Qsa[0][0,b] for b in range(9)]) for a in range(9)]
                norm_Q1_list = [self.Qsa[1][0,a] / np.max([self.Qsa[1][0,b] for b in range(9)]) for a in range(9)]
                print('Qsa[0][0,:]', [f'{q:.1e}' for q in norm_Q0_list])
                print('Qsa[1][0,:]', [f'{q:.1e}' for q in norm_Q1_list])
                print('Exproration: Epsilon=', self.epsilon)
        return cnt_trace        


def plot_cnt_trace(cnt_trace):
    N_cnt = len(cnt_trace)
    cnt_d = {'Equal':np.zeros(N_cnt,dtype=int), 
            'P1':np.zeros(N_cnt,dtype=int), 
            'P2':np.zeros(N_cnt,dtype=int)}
    for i, cnt in enumerate(cnt_trace):
        cnt_d['Equal'][i] = cnt[0]
        cnt_d['P1'][i] = cnt[1]
        cnt_d['P2'][i] = cnt[2]
    plt.plot(range(N_cnt), cnt_d['Equal'], label='Equal')
    plt.plot(range(N_cnt), cnt_d['P1'], label='Player1 wins')
    plt.plot(range(N_cnt), cnt_d['P2'], label='Player2 wins')
    plt.xlabel('Episode')
    plt.ylabel('Count')
    plt.legend(loc=0)
    plt.title('First and second playing agent performance in learning')
    plt.show()


def plot_cnt_trace_normal(cnt_trace_from0, title=''):
    """We ignore cnt_trace[0] since all values there are zero.
    It makes unable to calculate divide by sum, whch is sum of all zeros.
    """
    cnt_trace = cnt_trace_from0[1:]
    N_cnt = len(cnt_trace)
    cnt_d = {'Equal':np.zeros(N_cnt,dtype=float), 
            'P1':np.zeros(N_cnt,dtype=float), 
            'P2':np.zeros(N_cnt,dtype=float)}

    for i, cnt in enumerate(cnt_trace):
        sum_cnt = np.sum(cnt[0:3])
        cnt_d['Equal'][i] = cnt[0] / sum_cnt
        cnt_d['P1'][i] = cnt[1] / sum_cnt
        cnt_d['P2'][i] = cnt[2] / sum_cnt

    plt.plot(range(N_cnt), cnt_d['Equal'], label='Equal')
    plt.plot(range(N_cnt), cnt_d['P1'], label='Player1 wins')
    plt.plot(range(N_cnt), cnt_d['P2'], label='Player2 wins')
    plt.xlabel('Episode')
    plt.ylabel('Count')
    plt.legend(loc=0)
    plt.title('Normalized learning performance: ' + title)
    plt.show()


def plot_cnt_trace_normal_order(cnt_trace_from0, title=''):
    """We ignore cnt_trace[0] since all values there are zero.
    It makes unable to calculate divide by sum, whch is sum of all zeros.
    """
    cnt_trace = cnt_trace_from0[1:]
    N_cnt = len(cnt_trace)
    cnt_d = {'Equal':np.zeros(N_cnt,dtype=float), 
            'P1':np.zeros(N_cnt,dtype=float), 
            'P2':np.zeros(N_cnt,dtype=float),
            'PO1':np.zeros(N_cnt,dtype=float), 
            'PO2':np.zeros(N_cnt,dtype=float)}

    for i, cnt in enumerate(cnt_trace):
        sum_cnt = np.sum(cnt[0:3])
        # sum-cnt is equal to np.sum(cnt[0] + cnt[3:5])
        cnt_d['Equal'][i] = cnt[0] / sum_cnt
        cnt_d['P1'][i] = cnt[1] / sum_cnt
        cnt_d['P2'][i] = cnt[2] / sum_cnt
        cnt_d['PO1'][i] = cnt[3] / sum_cnt
        cnt_d['PO2'][i] = cnt[4] / sum_cnt

    plt.plot(range(N_cnt), cnt_d['Equal'], label='Equal')
    plt.plot(range(N_cnt), cnt_d['P1'], label='Player 1')
    plt.plot(range(N_cnt), cnt_d['P2'], label='Player 2')
    plt.plot(range(N_cnt), cnt_d['PO1'], label='Play order 1')
    plt.plot(range(N_cnt), cnt_d['PO2'], label='Play order 2')
    plt.grid()
    plt.xlabel('Episode')
    plt.ylabel('Normalized Counts')
    plt.legend(loc=0)
    plt.title('Normalized win counts: ' + title)
    plt.show()

def plot_cnt_trace_normal_order_detail(cnt_trace_from0, title=''):
    """Detail analysis is applied so that player 1 (our agent) performance history is divided it play first and second. 
    """
    cnt_trace = cnt_trace_from0[1:]
    N_cnt = len(cnt_trace)
    # N_types = len(cnt_trace[0])
    cnt_trace_a = np.array(cnt_trace, dtype=float)
    cnt_trace_sum = np.sum(cnt_trace_a[:,0:3], axis=1, keepdims=True)
    print(cnt_trace_a.shape, cnt_trace_sum.shape)
    cnt_trace_a = cnt_trace_a / cnt_trace_sum

    label_list = ['Tie', 'Player1', 'Player2', 'Order1', 'Order2', 'P1O1', 'P1O2', 'P2O1', 'P2O2']
    plt.figure(figsize=(16,7))
    plt.subplot(1,2,1)
    for i in range(5):
        plt.plot(range(N_cnt), cnt_trace_a[:,i], label=label_list[i])
    plt.grid()
    plt.xlabel('Episode')
    plt.ylabel('Normalized Counts')
    plt.legend(loc=0)
    plt.title('Normalized win counts: ' + title)

    plt.subplot(1,2,2)
    for i in [0,5,6,7,8]:
        plt.plot(range(N_cnt), cnt_trace_a[:,i], label=label_list[i])    
    plt.grid()
    plt.xlabel('Episode')
    plt.ylabel('Normalized Counts')
    plt.legend(loc=0)
    plt.title('Normalized win counts: ' + title)

    plt.show()


def learning_stage_mc(N_episodes=100, save_flag=True, fig_flag=False):
    ff = 0.9
    lr = 0.01
    N_Symbols = 3 # 0=empty, 1=plyaer1, 2=player2
    N_A = 9 # (0,0), (0,1), ..., (2,2)
    print_cnt = N_episodes / 10

    my_Q_System = Q_System(N_A, N_Symbols)
    cnt_trace = my_Q_System.learning(N_episodes=N_episodes, ff=ff, lr=lr, print_cnt=print_cnt)
    print('-------------------')
    cnt = cnt_trace[-1]
    print(N_episodes, cnt)

    if save_flag:
        my_Q_System.save()

    if fig_flag:
        plot_cnt_trace_normal(cnt_trace, 'MC')

    return my_Q_System


def generate_action_list_fn(action_list, N_A=3, stack=[]):
    for i in range(N_A):
        if i not in stack:
            stack.append(i)            
            # print(stack)
            action_list.append(stack.copy())
            # print(action_list)
            generate_action_list_fn(action_list, N_A, stack)
            stack.pop()


def generate_action_list(N_A=3): #N_A = 9 is for tictactoe
    action_list = []
    generate_action_list_fn(action_list, N_A=N_A)
    return action_list


def _generate_avaliable_action_list_fn(my_Q_System, action_list_collection, N_A=3, stack=[]):
    for i in range(N_A):
        if i not in stack:
            stack.append(i)            
            # print(stack)
            Buff = my_Q_System.play_by_scenario(1, action_list=stack)
            action_list_collection.append(stack.copy())
            if Buff['r'][-1] == 0:
                _generate_avaliable_action_list_fn(my_Q_System, action_list_collection, N_A, stack)
            else:
                print(stack, Buff['r'][-1])
            stack.pop()

def play_by_scenario_fn(action_list, N_A=9):
    """
    win_player = play_by_scenario_fn(stack)
    - Equivalent to self.play_by_scenario()

    Return win_player, which could be 1 or 2 depending on win player index
    """
    P_no = 1
    S = np.zeros((N_A,),dtype='int16')
    for action in action_list:
        # action, done = get_action_by_scenario()
        set_state_inplace(S, action, P_no)
        win_player = calc_reward_numba(MASK_L, S)
        if win_player:
            return win_player
        P_no = 1 if P_no == 2 else 2
    return 0


def action_list_2_state(action_list, N_A=9):
    """
    win_player = play_by_scenario_fn(stack)
    - Equivalent to self.play_by_scenario()

    Return win_player, which could be 1 or 2 depending on win player index
    """
    P_no = 1
    S = np.zeros((N_A,),dtype='int16')
    for action in action_list:
        # action, done = get_action_by_scenario()
        set_state_inplace(S, action, P_no)
        P_no = 1 if P_no == 2 else 2
    return S    


def play_by_scenario_fn_numba(action_list, N_A:int=9) -> int:
    """
    win_player = play_by_scenario_fn_numba(stack)
    - Equivalent to self.play_by_scenario()

    Return win_player, which could be 1 or 2 depending on win player index
    """
    P_no = 1
    S = np.zeros((N_A,),dtype='int16')
    for action in action_list:
        # action, done = get_action_by_scenario()
        set_state_inplace(S, action, P_no)
        win_player = calc_reward_numba(MASK_L, S)
        if win_player:
            return win_player
        P_no = 1 if P_no == 2 else 2
    return 0


def generate_avaliable_action_list_fn(action_list_collection, N_A=9, stack=[]):
    for i in range(N_A):
        if i not in stack:
            stack.append(i)            
            # print(stack)
            win_player = play_by_scenario_fn_numba(stack)
            action_list_collection.append(stack.copy())
            if win_player == 0:
                generate_avaliable_action_list_fn(action_list_collection, N_A, stack)
            stack.pop()


def generate_avaliable_action_list(): #N_A = 9 is for tictactoe
    """
    Generate all avaiable action list in the tictactoe game. 
    This is used for Bellman expectation equation.
    - Performing for the first playing case only. The second playing case will be considered later. 
    """
    N_A = 9
    action_list_collection = []
    generate_avaliable_action_list_fn(action_list_collection, N_A=N_A)
    return action_list_collection


def Bellman_expectation_fn(do_action, N_A=9, action_list=[]):
    for action in range(N_A):
        if action not in action_list:
            action_list.append(action)
            win_player = do_action.perform_action(action)            
            if win_player == 0:
                Bellman_expectation_fn(do_action, N_A, action_list)
            action_list.pop()

class DoAction:
    def __init__(self, N_A=9):
        self.player = 1
        self.S = np.zeros((N_A,),dtype=int)
        self.action_cnt = 0
        self.N_A = N_A
    def perform_action(self, action):
        self.action_cnt += 1
        set_state_inplace(self.S, action, self.player)
        win_player = calc_reward_numba(MASK_L, self.S)
        # Remind tictactoe has sparse reward. 

        if win_player != 0 or self.action_cnt == self.N_A:
            # no reward, yet
            pass
        else:
            # reward = ((-1 if win_player==2 else win_player) + 1)/2 # else -> lose
            pass
        self.player = 2 if self.player==1 else 1

def get_Vs_prt(Qsa, Saa, action_list, N_A=9, N_Symbols=3, disp_flag=False):
    Vs = 0.
    for action_pro in range(N_A):
        if action_pro not in action_list:
            Saa_Idx = calc_S_idx_numba(Saa, N_Symbols=N_Symbols)
            Vs += Qsa[Saa_Idx, action_pro]
            if disp_flag:
                print(f'Qsa[{Saa}, {action_pro}] = ', Qsa[Saa_Idx, action_pro])
    return Vs

def get_Vs(Qsa, Saa, action_list, N_A=9, N_Symbols=3):
    Vs = 0.
    for action_pro in range(N_A):
        if action_pro not in action_list:
            Saa_Idx = calc_S_idx_numba(Saa, N_Symbols=N_Symbols)
            Vs += Qsa[Saa_Idx, action_pro]
    return Vs


def Bellman_expectation(): #N_A = 9 is for tictactoe
    """
    Generate all avaiable action list in the tictactoe game. 
    This is used for Bellman expectation equation.
    - Performing for the first playing case only. The second playing case will be considered later. 
    """
    N_A = 9
    do_action = DoAction(N_A=N_A)
    Bellman_expectation_fn(do_action, N_A=N_A)


class Bellman:
    def __init__(self, N_A=9, N_Symbols=3, ff=0.9): 
        N_S = N_Symbols ** N_A
        self.Qsa = np.zeros((N_S, N_A))
        self.N_A = N_A
        self.N_Symbols = N_Symbols
        self.ff = ff
    
    def update(self, action_list=[], max_actions=8): 
        """This will update
        """
        # Bellman_exp_inplace(self.Qsa, S, action, action_list, ff=0.9, N_A=9, N_Symbols=3):
        
        # This is testing code
        # print('Top: action_list:', action_list)
        if len(action_list) > min(max_actions, self.N_A - 1):
            # action_list should be less than N_A - 1 since additional action will be considered to calaculate Qsa(S,a)    
            # this is already prohibited by Bellman_exp_inplace which provides done=True if len(action_list + action) == N_A
            # print('Max action stop: action_list:', action_list)
            return 
        
        if len(action_list) % 2 == 1: # a stage for player 2 to play
            for action in range(self.N_A):
                if action not in action_list:
                    action_list.append(action)
                    Sa = action_list_2_state(action_list)
                    done = False if calc_reward_numba(MASK_L, Sa) == 0 else True
                    if not done:
                        # print('Enter: action_list:', action_list)
                        self.update(action_list, max_actions=max_actions)
                    #else:
                    #   print('Player 2 wins with Sa, action_list and done', Sa, action_list, done)
                    action_list.pop()  
        else: # player 1
            S = action_list_2_state(action_list)
            for action in range(self.N_A):
                if action not in action_list:
                    done = Bellman_exp_inplace(self.Qsa, S, action, action_list, ff=self.ff, N_A=self.N_A, N_Symbols=self.N_Symbols)    
                    if not done:
                        action_list.append(action)
                        # print('Enter: action_list:', action_list)
                        self.update(action_list, max_actions=max_actions)
                        action_list.pop()

    def check(self, action_list=[0,3,1,4], action=2):
        S = action_list_2_state(action_list)
        S_Idx = calc_S_idx_numba(S, N_Symbols=self.N_Symbols)
        print(f'Qsa({S},{action})={self.Qsa[S_Idx][action]}')

def Bellman_exp_inplace(Qsa, S, action, action_list, ff=0.9, N_A=9, N_Symbols=3):
    P_no = 1    
    Sa = set_state(S, action, P_no)
    win_player = calc_reward_numba(MASK_L, Sa)
    action_list.append(action)
    # Notice that R could not be changed by player 2. It is only determined by player 1.
    done = False
    E_V_Saa = 0. # Saa -> S(t+1) for Player1 
    if win_player == P_no: # P_no <- 1
        E_R = 1.0
        done = True
    elif len(action_list) == N_A: # no more following action.
        E_R = 0.5
        done = True
    else:
        E_R = 0.
        # get v(S_t+1)
        # Here, pro means proponent
        P_no_pro = 2 if P_no == 1 else 1
        for action_pro in range(N_A): 
            if action_pro not in action_list:
                Saa = set_state(Sa, action_pro, P_no_pro)
                win_player_pro = calc_reward_numba(MASK_L, Saa)
                action_list.append(action_pro)                
                if win_player_pro != P_no_pro and len(action_list) != N_A:
                    D_E_V_Saa = get_Vs(Qsa, Saa, action_list, N_A=N_A, N_Symbols=N_Symbols)
                    E_V_Saa += D_E_V_Saa
                action_list.pop()
    action_list.pop()
    S_Idx = calc_S_idx_numba(S, N_Symbols=N_Symbols)
    Qsa[S_Idx, action] = E_R + ff*E_V_Saa
    return done


def Bellman_exp_inplace_prt(Qsa, S, action, action_list, ff=0.9, N_A=9, N_Symbols=3, disp_flag=False):
    """Evaluate q(s,a) using ``S`` and ``action``.    

    Returns q(s,a).

    Parameters
    ----------
    S : numpy.ndarray, shape=(N_A,)
        State matrix <--> state index, ``S_Idx``
    action : int
        action value in [0, 1, ..., N_A]

    Returns
    -------
    q(s,a): numpy.ndarray, shape=(N_S, N_A)
            q-value w.r.t. S and action

    Notes
    -----
    This function calls Bellman_exp_fn.

    References
    ----------
    .. [1] t.b.d.
           Retrieved from https://t.b.d

    Examples
    --------
    Exmaples will be described.

    >>> import tictactoe # below will be updated. 
    >>> x = np.array([[1., 2., 3.],
    ...               [4., 5., 6.],
    ...               [7., 8., 9.]])
    >>> y = np.array([[1., 2., 3.],
    ...               [4., 5., 6.]])
    >>> pairwise_dists(x, y)
    array([[ 0.        ,  5.19615242],
           [ 5.19615242,  0.        ],
           [10.39230485,  5.19615242]])
    """
    P_no = 1    
    Sa = set_state(S, action, P_no)
    print('-----------------------')
    print('S:', S, ' with action: ', action, ' based on ', action_list)
    print('Sa:', Sa)
    if disp_flag:
        print('[Saa]')
    win_player = calc_reward_numba(MASK_L, Sa)
    action_list.append(action)
    # Notice that R could not be changed by player 2. It is only determined by player 1.
    done = False
    E_V_Saa = 0. # Saa -> S(t+1) for Player1 
    if win_player == P_no: # P_no <- 1
        E_R = 1.0
        done = True
    elif len(action_list) == N_A: # no more following action.
        E_R = 0.5
        done = True
    else:
        E_R = 0.
        # get v(S_t+1)
        # Here, pro means proponent
        P_no_pro = 2 if P_no == 1 else 1
        for action_pro in range(N_A): 
            if action_pro not in action_list:
                Saa = set_state(Sa, action_pro, P_no_pro)
                win_player_pro = calc_reward_numba(MASK_L, Saa)
                if disp_flag:
                    print(Saa)
                    print(f'win_player_pro = {win_player_pro}')
                action_list.append(action_pro)                
                if win_player_pro != P_no_pro and len(action_list) != N_A:
                    D_E_V_Saa = get_Vs_prt(Qsa, Saa, action_list, N_A=N_A, N_Symbols=N_Symbols, disp_flag=disp_flag)
                    E_V_Saa += D_E_V_Saa
                    if disp_flag:
                        print(f'D_E_V_Saa = {D_E_V_Saa}')
                action_list.pop()
    action_list.pop()
    S_Idx = calc_S_idx_numba(S, N_Symbols=N_Symbols)
    Qsa[S_Idx, action] = E_R + ff*E_V_Saa
    print('[Returns]')
    print(f'E_R = {E_R}, E_V_Saa = {E_V_Saa}')
    return done

def calc_Bellman_exp(action=4, action_list=[]):
    ff = 0.9
    N_A = 9
    N_Symbols = 3 # 0, 1, 2 (represent empty, player1 stone, player2 stone, respectively)
    N_S = N_Symbols ** N_A
    # S = np.zeros((N_A,), dtype=int)
    Qsa = np.zeros((N_S, N_A))

    # action_list = []
    # action = 4    
    S = action_list_2_state(action_list)
    Bellman_exp_inplace(Qsa, S, action, action_list, ff=ff, N_A=N_A, N_Symbols=N_Symbols)
    
    S_idx = calc_S_idx_numba(S, N_Symbols=N_Symbols)
    print('[Result]')
    print(f'Qsa({S}, {action}) -> ', Qsa[S_idx, action])


def calc_total_states(N_A=9):
    N = 1
    for i in range(N_A):
        # print(range(N_A-i,N_A+1), np.prod(range(N_A-i,N_A+1)))
        N += np.prod(range(N_A-i,N_A+1))
    return N

def input_default_with(str, defalut_value, dtype=int):
    answer = input(str+f"[default={defalut_value}] ")
    if answer == '':
        return defalut_value
    else:
        return dtype(answer)

def input_default(str, defalut_value, dtype=int):
    answer = input(str)
    if answer == '':
        return defalut_value
    else:
        return dtype(answer)

def check_play_by_scenario():
    ff = 0.9
    lr = 0.01
    N_Symbols = 3 # 0=empty, 1=plyaer1, 2=player2
    N_A = 9 # (0,0), (0,1), ..., (2,2)
    my_Q_System =  Q_System(N_A, N_Symbols)

    Buff = my_Q_System.play_by_scenario(
        P_no=1, action_list=[4,1,3,5,6,2,0])
    print('---------------------------')
    print('Buff')
    print(Buff)

    Buff_dual= buff_depart(Buff)
    print('---------------------------')
    print('Buff_dual')
    print(Buff_dual)

    player = 1
    discounted_inplace(Buff_dual[player-1]['r'], ff)
    print('---------------------------')
    print("Buff_dual[player-1]['r'] with player=1")
    print(Buff_dual[player-1]['r'])       

    my_Q_System.disp_flag = True
    my_Q_System.updateQsa_inplace(my_Q_System.Qsa[player-1], Buff_dual[player-1], lr)


class Testing:
    def __init__(self, fn_name):
        '''Usages:
            - Testing('calc_reward_tf')
        '''
        if fn_name == 'calc_reward_tf':
            self.test_calc_reward_tf()
        elif fn_name == 'find_action_list':
            self.test_find_action_list()
        elif fn_name == 'get_action':
            self.test_get_action()
        elif fn_name == 'all':
            self.test_calc_reward_tf()
            self.test_find_action_list()
            self.test_get_action()
    
    def test_calc_reward_tf(self):
        S_examples = tf.constant([[0,0,0, 0,0,0, 0,0,0],
                                 [1,1,1, 2,0,2, 2,0,0],
                                 [0,0,2, 1,2,1, 2,0,0]])
        
        print('===================================')
        print('Testing: calc_reward_tf')
        print('[Anwer]')
        answer = [0, 1, 2]
        print(answer)

        print('-------------------------------------')
        print('[Test]')
        test = [calc_reward_numba(MASK_L, S) for S in S_examples] 
        print(test)  
        if test == answer:
            print('Test Ok')
        else:
            print('Test fail')
            
    def test_find_action_list(self):
        print('===================================')
        print('Testing: test_find_action_list')
        print('[Answer]')
        print('''[[0 0 0]
 [0 0 0]
 [0 0 0]] [0, 1, 2, 3, 4, 5, 6, 7, 8] 0
[[0 2 0]
 [0 1 0]
 [1 0 2]] [0, 2, 3, 5, 7] 4''')
        
        N_A = 9
        N_Symbols = 3 
        my_Q_System = Q_System(N_A, N_Symbols)

        print('-------------------------------------')
        print('[Test]')
        S_l = [[0,0,0, 0,0,0, 0,0,0], [0,2,0, 0,1,0, 1,0,2]]
        for S in S_l:
            action_list, no_occupied = my_Q_System.find_action_list(S)
            print(np.reshape(S,(3,3)), action_list, no_occupied)
            
    def test_get_action(self):
        print('===================================')
        print('Testing: get_action')        
        print('''[Answer]
Equally max actions: [0]
S_idx 0 action 0 action_list [0, 1, 2, 3, 4, 5, 6, 7, 8] action_prob [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
[[0 0 0]
 [0 0 0]
 [0 0 0]] 1 0
Equally max actions: [0]
S_idx 13950 action 0 action_list [0, 1, 3, 5, 7] action_prob [1.0, 0.0, 0.0, 0.0, 0.0]
[[0 0 2]
 [0 1 0]
 [1 0 2]] 1 0
Equally max actions: [0]
S_idx 0 action 0 action_list [0, 1, 2, 3, 4, 5, 6, 7, 8] action_prob [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
[[0 0 0]
 [0 0 0]
 [0 0 0]] 2 0
Equally max actions: [0]
S_idx 13950 action 0 action_list [0, 1, 3, 5, 7] action_prob [1.0, 0.0, 0.0, 0.0, 0.0]
[[0 0 2]
 [0 1 0]
 [1 0 2]] 2 0''')
        N_A = 9
        N_Symbols = 3 
        my_Q_System = Q_System(N_A, N_Symbols)

        print('-------------------------------------')
        print('[Test]')
        S_l = [[0,0,0, 0,0,0, 0,0,0], [0,0,2, 0,1,0, 1,0,2]]
        for P_no in [1,2]:
            for S in S_l:
                S_idx = my_Q_System.calc_S_idx(S)
                my_Q_System.Qsa[P_no-1][S_idx,:] = np.array([1.0,0.0,0, 0,0,0, 0,0,0])    
                action, _ = my_Q_System.get_action(P_no, S)
                print(np.reshape(S,(3,3)), P_no, action)


def _main():
    ff = 0.9
    lr = 0.01
    N_episodes = 2
    N_Symbols = 3 # 0=empty, 1=plyaer1, 2=player2
    N_A = 9 # (0,0), (0,1), ..., (2,2)

    my_Q_System = Q_System(N_A, N_Symbols)
    cnt = [0, 0, 0] # tie, p1, p2
    player = 1
    for episode in range(N_episodes):
        # print('===================================')
        # Can save this data for play 2 as well
        Buff = my_Q_System.play(player)    
        my_Q_System.update_Qsa_inplace(Buff, ff=ff, lr=lr)    
        win_player = 0 if Buff['r'][-1] == 0 else Buff['P_no'][-1]
        cnt[win_player] += 1

        player = 2 if player == 1 else 1
        if episode % 10 == 0:
            print(episode, cnt)

    print(cnt)


# preparing a new environment for tictactoc based on gym
# Since gym is standard, the tictactoe game environment is developed based on gym.
from random import sample

class Tictactoe_Env:
    def __init__(self, N_A:int=9, play_order:int=1):
        """
        Inputs
        ------        
        play_order: 1 or 2
                    the order for playing. It should be 1 or 2 since tictactoe is a two player game.

        Fix P_no = 1 since our agent is assumed to be player 1 regardless of its playing order.
        """
        self.play_order = play_order
        self.N_A = N_A
        self.S = []
        self.action_list = []
        self.P_no = 1
        self.P_no_opponent = 2

    def get_player_action(self):
        return self.sample_action()

    def step(self, action):
        """S.copy() and action_list.copy() are returned so that internal states 
        shuould not be changed.
        """
        done = False
        if action not in self.action_list:
            self.action_list.append(action)
            set_state_inplace(self.S, action, self.P_no)
            win_player = calc_reward_numba(MASK_L, self.S)
            if win_player == self.P_no:
                reward = 1.0
                done = True
            elif len(self.action_list) == self.N_A: # Tie
                reward = 0.5
                done = True
            else:
                self.perform_player2()
                win_player = calc_reward_numba(MASK_L, self.S)
                if win_player == self.P_no_opponent:
                    reward = 0.0
                    done = True
                elif len(self.action_list) == self.N_A: # Tie 
                    reward = 0.5
                    done = True
                else: # Continue playing
                    reward = 0.0

        # By copying this information, S and action_list, 
        # self.S, self.action_list can be managed separtely and safely
        S_copy = self.S.copy()
        action_list_copy = self.action_list.copy() 
        if done:
            self.reset()
        return S_copy, action_list_copy, reward, done

    def step_op(self, action):
        """S.copy() and action_list.copy() are returned so that internal states 
        shuould not be changed.
        """
        op_action = None # Because of done
        done = False
        if action not in self.action_list:
            self.action_list.append(action)
            set_state_inplace(self.S, action, self.P_no)
            win_player = calc_reward_numba(MASK_L, self.S)
            if win_player == self.P_no:
                reward = 1.0
                done = True
            elif len(self.action_list) == self.N_A: # Tie
                reward = 0.5
                done = True
            else:
                op_action = self.perform_player2_op()
                win_player = calc_reward_numba(MASK_L, self.S)
                if win_player == self.P_no_opponent:
                    reward = 0.0
                    done = True
                elif len(self.action_list) == self.N_A: # Tie 
                    reward = 0.5
                    done = True
                else: # Continue playing
                    reward = 0.0

        S_copy = self.S.copy()
        action_list_copy = self.action_list.copy() 
        if done:
            self.reset()
        return S_copy, action_list_copy, reward, done, op_action


    def _perform_player2(self):
        action = self.get_player_action()
        self.action_list.append(action)
        set_state_inplace(self.S, action, self.P_no_opponent)

    def perform_player(self, P_no):
        """
        perform by player P_no
        """
        action = self.get_player_action()
        self.action_list.append(action)
        set_state_inplace(self.S, action, P_no)    

    def perform_player_op(self, P_no):
        """
        perform by player P_no
        """
        action = self.get_player_action()
        self.action_list.append(action)
        set_state_inplace(self.S, action, P_no)    
        return action

    def perform_player2(self):
        self.perform_player(self.P_no_opponent)

    def perform_player2_op(self):
        return self.perform_player_op(self.P_no_opponent)

    def _reset(self, play_order=None): # consider play_order (now we assume play_order=1)
        if play_order is not None:
            self.play_order = play_order
        self.S = np.zeros((self.N_A,),dtype=int)
        self.action_list = []
        if self.play_order == 2:
            self.perform_player2()
        # return current results. we use value sharing but not buff
        return self.S.copy(), self.action_list.copy()

    def reset(self, play_order=None): # consider play_order (now we assume play_order=1)
        """
        Make S as list, not array
        Check S type for the first in this class
        """
        if play_order is not None:
            self.play_order = play_order
        self.S = np.zeros((self.N_A,),dtype=int)
        self.action_list = []
        if self.play_order == 2:
            self.perform_player2()
        # return current results. we use value sharing but not buff
        return self.S.copy(), self.action_list.copy()

    def render(self, mode='human'):
        print('Current state')
        print(self.S.reshape(-1,3))

    def sample_action(self):
        """Sample action not in action list for both player 1 and 2,
        i.e., regardless of player index
        """
        actions = []
        for action in range(self.N_A):
            if action not in self.action_list:
                actions.append(action)
        return sample(actions,1)[0]


class Tictactoe_Env_Active(Tictactoe_Env):
    def __init__(self, N_A:int=9, play_order:int=1, agent_type:int=0, Q_System_Class=Q_System):
        """
        agent_type = 0(random), 1(advanced)
        """
        super(Tictactoe_Env_Active,self).__init__(N_A=N_A, play_order=play_order)
        
        self.agent_type = agent_type
        self.q_sys = Q_System_Class(N_A=N_A, N_Symbols=3)
        self.q_sys.load()

    def advanced_action(self):
        P_no = 1
        action, _ = self.q_sys.get_action(P_no, self.S)
        return action

    def premium_action(self):
        P_no = 1
        action, _ = self.q_sys.get_action_against_human(P_no, self.S)
        return action       

    def get_player_action(self):
        if self.agent_type == 0: # random
            return self.sample_action()
        elif self.agent_type == 1: # #1, advanced
            return self.advanced_action()
        else: # agent_type == 2, premium
            return self.premium_action()


def test_tictactoe_env(play_order:int=1, disp_flag:bool=False) -> float:
    N_A = 9
    ttt_env = Tictactoe_Env(N_A, play_order=play_order) #both X but start 1st and 2nd

    ttt_env.reset()
    for _ in range(10):
        if disp_flag:
            ttt_env.render()
        S, action_list, reward, done = ttt_env.step(ttt_env.sample_action()) # take a random action
        if done:
            if disp_flag:
                print('Last state:')
                print(S.reshape(-1,3))
                print(f'action_list={action_list}, reward={reward}')
            break

    return reward

def multiple_test_tictactoe_env(N:int=10, play_order:int=1, disp_flag:bool=False) -> float:
    reward_a = np.zeros((N,))
    for i in range(N):
        reward_a[i] = test_tictactoe_env(play_order=play_order,disp_flag=disp_flag)
    if disp_flag:
        print(f'reward_a={reward_a}')
        print(f'average reward={np.average(reward_a)}')
    return np.average(reward_a)


def _main():
    Q1 = input_default('1. Loading a trained agent (0) or Learning a new agent (1)? (default=0) ', 0, int)
    if Q1 == 0:
        print('Loading the trained agent...')
        Q2 = input_default('2. Do you want to start first?(0=yes,1=no,default=0) ', 0, int)
        player_human = Q2 + 1
        if player_human == 1:
            print('You=1(X), Agent=2(O)') 
        else:
            print('Agent=1(X), You=2(O)') 
        trained_Q_System = Q_System(None)
        trained_Q_System.load()
        trained_Q_System.play_with_human_inference(player_human)
        # print(len(trained_Q_System.Qsa))
    else:
        print('Start to learn a new agent...')
        Q2 = input_default('2. How many episode do you want to learn?(default=10000) ', 10000, int)
        # my_Q_System = learning_stage(N_episodes=Q2, fig_flag=True)
        _ = learning_stage_mc(N_episodes=Q2, fig_flag=True)
        # print(len(my_Q_System.Qsa))

class Q_System_QL(Q_System):
    def __init__(self, N_A, N_Symbols):
        super(Q_System_QL, self).__init__(N_A=N_A, N_Symbols=N_Symbols)

    def q_learning_batch(self, Replay_buff_d, P_no=1, ff=0.9, lr=0.01):
        """q_learning_batch(self, Replay_buff_d, P_no=1, ff=0.9, lr=0.01)
        
        Because of shuffling, (1-d) should be considered for y calculation (or td_err).
        """
        random_shuffle_dict_inplace(Replay_buff_d) # need to be updated to consider each list
        for j in range(len(Replay_buff_d['reward'])):
            S = Replay_buff_d['S'][j]
            action = Replay_buff_d['action'][j]
            S_new = Replay_buff_d['S_new'][j]
            reward = Replay_buff_d['reward'][j]
            done_int = Replay_buff_d['done'][j]
            S_new_idx = calc_S_idx_numba(S_new, self.N_Symbols)
            S_idx = calc_S_idx_numba(S, self.N_Symbols)
            # if done, no future reward is generated.
            y = reward + (1-done_int) * ff * np.max(self.Qsa[P_no-1][S_new_idx,:]) # ff --> (1-d)*ff should be updated
            td_err = y - self.Qsa[P_no-1][S_idx, action]
            self.Qsa[P_no-1][S_idx, action] += lr * td_err


    def q_learning_planning_batch(self, env_model, Replay_buff_d, N_plan=1, P_no=1, ff=0.9, lr=0.01):
        for _ in range(N_plan):
            random_shuffle_dict_inplace(Replay_buff_d)
            generated_Replay_buff_d = env_model.sampling(Replay_buff_d, P_no)
            self.q_learning_batch(generated_Replay_buff_d, P_no=P_no, ff=ff, lr=lr)

    def _learning(self, N_episodes=2, ff=0.9, lr=0.01, print_cnt=10):
        """Return: 
            cnt_trace = [cnt, ...]: cnt vector are stacked in cnt_trace
        """
        play_order = 1
        ttt_env = Tictactoe_Env(self.N_A, play_order=play_order) #both X but start 1st and 2nd

        cnt = [0, 0, 0] # tie, p1, p2
        cnt_trace = [cnt.copy()]        
        player = play_order
        P_no = player

        for episode in range(N_episodes):
            S, _ = ttt_env.reset()
            done = False            
            Replay_buff = []
            while not done:
                self.epsilon = 0.1 # epsilon is a hyperparamter for exploration
                action, _ = self.get_action(P_no, S)
                S_new, _, reward, done = ttt_env.step(action)
                Replay_buff.append([S, action, S_new, reward])
                # print(episode, [S, action, S_new, reward])
                S = S_new

            if Replay_buff[-1][3] == 1.0:
                cnt[1] += 1   
            elif Replay_buff[-1][3] == 0.5:
                cnt[0] += 1   
            else:
                cnt[2] += 1

            cnt_trace.append(cnt.copy())

            if episode % print_cnt == 0:
                print(episode, cnt)                
                print('S = [0,0,0, 0,0,0, 0,0,0]')
                print('Qsa[0][0,:]', [f'{self.Qsa[0][0,a]:.1e}' for a in range(9)])
                print('Qsa[1][0,:]', [f'{self.Qsa[1][0,a]:.1e}' for a in range(9)])
                print('Exproration: Epsilon=', self.epsilon)
        return cnt_trace

    def _learning(self, N_episodes=2, ff=0.9, lr=0.01, epsilon = 0.4, print_cnt=10):
        """Return: 
            cnt_trace = [cnt, ...]: cnt vector are stacked in cnt_trace
        """
        cnt = [0, 0, 0, 0, 0] # tie, p1, p2
        cnt_trace = [cnt.copy()]        

        # Opponent player index
        P_no = 1 # player Q function, regardless of play order (first or next)
        play_order = 1
        ttt_env = Tictactoe_Env(self.N_A, play_order=play_order) #both X but start 1st and 2nd
        for episode in range(N_episodes):
            S, _ = ttt_env.reset(play_order=play_order)
            done = False            
            Replay_buff = []
            while not done:
                self.epsilon = epsilon # epsilon is a hyperparamter for exploration
                action, _ = self.get_action(P_no, S)
                S_new, _, reward, done = ttt_env.step(action)
                Replay_buff.append([S.copy(), action, S_new.copy(), reward])
                # print(episode, [S, action, S_new, reward])
                S = S_new

            #######################################
            # DQN start, here for learning   
            #######################################
            # print('play_order, P_no = ', play_order, P_no)

            if Replay_buff[-1][3] == 1.0: 
                cnt[1] += 1               # play_order = 1 
                cnt[2 + play_order] += 1  # P_no = 1 (first player)
            elif Replay_buff[-1][3] == 0.5:
                cnt[0] += 1   
                cnt[2 + 3 - play_order] += 1  # P_no = 2 (second player)
            else: # play_order = 2
                cnt[2] += 1

            cnt_trace.append(cnt.copy())

            if episode % print_cnt == 0:
                print(episode, cnt)                
                print('S = [0,0,0, 0,0,0, 0,0,0]')
                print('Qsa[0][0,:]', [f'{self.Qsa[0][0,a]:.1e}' for a in range(9)])
                print('Qsa[1][0,:]', [f'{self.Qsa[1][0,a]:.1e}' for a in range(9)])
                print('Exproration: Epsilon=', self.epsilon)

            random.shuffle(Replay_buff) # inplace shuffling
            # y = np.zeros((len(Replay_buff),))
            for j in range(len(Replay_buff)):
                buff_each  =  Replay_buff[j]
                S, action, S_new, reward = buff_each
                S_new_idx = calc_S_idx_numba(S_new, self.N_Symbols)
                y = reward + ff * np.max(self.Qsa[P_no-1][S_new_idx,:])
                S_idx = calc_S_idx_numba(S, self.N_Symbols)
                self.Qsa[P_no-1][S_idx, action] += lr * y

            play_order = 3 - play_order # 1 --> 2, 2 --> 1

        return cnt_trace

    def learning(self, N_episodes=2, ff=0.9, lr=0.01, epsilon = 0.4, print_cnt=10):
        """Return: 
            cnt_trace = [cnt, ...]: cnt vector are stacked in cnt_trace
        """
        cnt = [0, 0, 0, 0, 0, 0, 0, 0, 0] # tie, p1, p2
        cnt_trace = [cnt.copy()]        

        # Opponent player index
        P_no = 1 # player Q function, regardless of play order (first or next)
        play_order = 1
        ttt_env = Tictactoe_Env(self.N_A, play_order=play_order) #both X but start 1st and 2nd
        for episode in range(N_episodes):
            S, _ = ttt_env.reset(play_order=play_order)
            done = False            
            Replay_buff = []
            while not done:
                self.epsilon = epsilon # epsilon is a hyperparamter for exploration
                action, _ = self.get_action(P_no, S)
                S_new, _, reward, done = ttt_env.step(action)
                Replay_buff.append([S.copy(), action, S_new.copy(), reward])
                # print(episode, [S, action, S_new, reward])
                S = S_new

            #######################################
            # DQN start, here for learning   
            #######################################
            # print('play_order, P_no = ', play_order, P_no)

            if Replay_buff[-1][3] == 1.0: 
                cnt[1] += 1               # P_no = 1 
                cnt[2 + play_order] += 1  # player_order = 1
                cnt[4 + play_order] += 1  # player_order | P_no = 1, so it occupied 2 lists
            elif Replay_buff[-1][3] == 0.5:
                cnt[0] += 1 
            else: # play_order = 2
                cnt[2] += 1                   # P_no = 2  
                cnt[2 + 3 - play_order] += 1  # player_order = 2
                cnt[6 + 3 - play_order] += 1  # player_order | P_no = 2, so it start from 6

            cnt_trace.append(cnt.copy())

            if episode % print_cnt == 0:
                print(episode, cnt)                
                print('S = [0,0,0, 0,0,0, 0,0,0]')
                print('Qsa[0][0,:]', [f'{self.Qsa[0][0,a]:.1e}' for a in range(9)])
                print('Qsa[1][0,:]', [f'{self.Qsa[1][0,a]:.1e}' for a in range(9)])
                print('Exproration: Epsilon=', self.epsilon)

            random.shuffle(Replay_buff) # inplace shuffling
            # y = np.zeros((len(Replay_buff),))
            for j in range(len(Replay_buff)):
                buff_each  =  Replay_buff[j]
                S, action, S_new, reward = buff_each
                S_new_idx = calc_S_idx_numba(S_new, self.N_Symbols)
                y = reward + ff * np.max(self.Qsa[P_no-1][S_new_idx,:])
                S_idx = calc_S_idx_numba(S, self.N_Symbols)
                self.Qsa[P_no-1][S_idx, action] += lr * y

            play_order = 3 - play_order # 1 --> 2, 2 --> 1

        return cnt_trace        

    def learning_qlearn(self, N_episodes=2, ff=0.9, lr=0.01, epsilon = 0.4, print_cnt=10):
        """Return: 
            cnt_trace = [cnt, ...]: cnt vector are stacked in cnt_trace
        """
        cnt = [0, 0, 0, 0, 0, 0, 0, 0, 0] # tie, p1, p2
        cnt_trace = [cnt.copy()]        

        # Opponent player index
        P_no = 1 # player Q function, regardless of play order (first or next)
        play_order = 1
        ttt_env = Tictactoe_Env(self.N_A, play_order=play_order) #both X but start 1st and 2nd
        for episode in range(N_episodes):
            S, _ = ttt_env.reset(play_order=play_order)
            done = False            
            aReplay_buff = ReplayBuff()
            while not done:
                self.epsilon = epsilon # epsilon is a hyperparamter for exploration
                action, _ = self.get_action(P_no, S)
                S_new, _, reward, done = ttt_env.step(action)

                # Save to replay buffers
                aReplay_buff.append(S.copy(), action, S_new.copy(), reward, done)
                
                # print(episode, [S, action, S_new, reward])
                S = S_new

            #######################################
            # DQN start, here for learning   
            #######################################
            # print('play_order, P_no = ', play_order, P_no)

            if aReplay_buff.d['reward'][-1] == 1.0: 
                cnt[1] += 1               # P_no = 1 
                cnt[2 + play_order] += 1  # player_order = 1
                cnt[4 + play_order] += 1  # player_order | P_no = 1, so it occupied 2 lists
            elif aReplay_buff.d['reward'][-1] == 0.5:
                cnt[0] += 1 
            else: # play_order = 2
                cnt[2] += 1                   # P_no = 2  
                cnt[2 + 3 - play_order] += 1  # player_order = 2
                cnt[6 + 3 - play_order] += 1  # player_order | P_no = 2, so it start from 6

            cnt_trace.append(cnt.copy())

            if episode % print_cnt == 0:
                print(episode, cnt)                
                print('S = [0,0,0, 0,0,0, 0,0,0]')
                print('Qsa[0][0,:]', [f'{self.Qsa[0][0,a]:.1e}' for a in range(9)])
                print('Qsa[1][0,:]', [f'{self.Qsa[1][0,a]:.1e}' for a in range(9)])
                print('Exproration: Epsilon=', self.epsilon)

            # Q-learning
            self.q_learning_batch(aReplay_buff.d, P_no=P_no, ff=ff, lr=lr)            

            play_order = 3 - play_order # 1 --> 2, 2 --> 1

        return cnt_trace

    def _modeling_qlearn(self, N_episodes=2, disp_flag=False):
        """
        Modeling of the game system is performed. 

        Return
        ------
        Not implemented yet.
        """        
        S_idx_base = np.power(self.N_Symbols,range(self.N_A))
        All_S = np.power(self.N_Symbols,self.N_A)
        # Model: All states x all actions x all associated new states x (reward, prob)
        self.Model_P = np.zeros((All_S, self.N_A, self.N_A), dtype=np.float32) #(Reward, p)
        self.Model_R = np.zeros((All_S, self.N_A), dtype=np.float32) #(Reward, p)
        self.N_Model = np.zeros((All_S, self.N_A), dtype=np.int) #(Reward, p)
        # Opponent player index
        P_no = 1 # player Q function, regardless of play order (first or next)
        play_order = 1
        ttt_env = Tictactoe_Env(self.N_A, play_order=play_order) #both X but start 1st and 2nd
        for _ in range(N_episodes):
            S, _ = ttt_env.reset(play_order=play_order)
            done = False            
            while not done:
                action, _ = self.get_action(P_no, S)
                S_new, _, reward, done, op_action = ttt_env.step_op(action)
                S_idx = np.sum(S_idx_base * S)
                R = self.Model_R[S_idx, action]
                N = self.N_Model[S_idx, action]
                self.Model_R[S_idx, action] = (R*N + reward)/(N+1)
                self.N_Model[S_idx, action] = N + 1
                if op_action is not None:
                    P = self.Model_P[S_idx, action, op_action]
                    self.Model_P[S_idx, action, op_action] = (P*N + 1)/(N+1)

                if disp_flag:
                    print(f'S_idx, action, op_action = {S_idx}, {action}, {op_action}')
                    print(f'Model_R, N_Model =',
                            self.Model_R[S_idx, action],
                            self.N_Model[S_idx, action])
                    if op_action is not None:
                        print(f'Model_P =',
                                self.Model_P[S_idx, action, op_action])

                S = S_new
            play_order = 1 if play_order == 1 else 2            

    def get_Model_P(self, S_idx, action, op_action):
        if self.N_Model[S_idx, action]:
            return self.Model_P_N[S_idx, action, op_action] / self.N_Model[S_idx, action]
        else:
            return 0.0

    def get_Model_R(self, S_idx, action):
        if self.N_Model[S_idx, action]:
            return self.Model_R_N[S_idx, action] / self.N_Model[S_idx, action]
        else:
            return 0.0

    def get_Model_Done1(self, S_idx, action):
        if self.N_Model[S_idx, action]:
            return self.Model_Done1_N[S_idx, action] / self.N_Model[S_idx, action]
        else:
            return 0.0            

    def _get_Model_Done2(self, S_idx, action, op_action):
        if self.N_Model[S_idx, action]:
            return self.Model_Done2_N[S_idx, action, op_action] / self.N_Model[S_idx, action]
        else:
            return 0.0

    def get_Model_Done2(self, S_idx, action, op_action):
        if self.Model_Done1_N[S_idx, action]:
            return self.Model_Done2_N[S_idx, action, op_action] / self.Model_Done1_N[S_idx, action] 
        else:
            return 0.0

    def _modeling_qlearn(self, N_episodes=2, disp_flag=False):
        """
        Modeling of the game system is performed. 

        Return
        ------
        Not implemented yet.
        """        
        S_idx_base = np.power(self.N_Symbols,range(self.N_A))
        All_S = np.power(self.N_Symbols,self.N_A)
        # Model: All states x all actions x all associated new states x (reward, prob)
        self.Model_P_N = np.zeros((All_S, self.N_A, self.N_A), dtype=np.float32) #(Reward, p)
        self.Model_R_N = np.zeros((All_S, self.N_A), dtype=np.float32) #(Reward, p)
        self.N_Model = np.zeros((All_S, self.N_A), dtype=np.int) #(Reward, p)
        # Opponent player index
        P_no = 1 # player Q function, regardless of play order (first or next)
        play_order = 1
        ttt_env = Tictactoe_Env(self.N_A, play_order=play_order) #both X but start 1st and 2nd
        for _ in range(N_episodes):
            S, _ = ttt_env.reset(play_order=play_order)
            done = False            
            while not done:
                action, _ = self.get_action(P_no, S)
                S_new, _, reward, done, op_action = ttt_env.step_op(action)
                S_idx = np.sum(S_idx_base * S)

                self.Model_R_N[S_idx, action] += reward
                self.N_Model[S_idx, action] += 1
                if op_action is not None:
                    self.Model_P_N[S_idx, action, op_action] += 1.0

                if disp_flag:
                    print(f'S_idx, action, op_action = {S_idx}, {action}, {op_action}')
                    print(f'Model_R, N_Model =',
                            self.Model_R_N[S_idx, action],
                            self.N_Model[S_idx, action])
                    if op_action is not None:
                        print(f'Model_P =',
                                self.Model_P_N[S_idx, action, op_action])

                S = S_new
            play_order = 1 if play_order == 1 else 2

    def modeling_tbl_init(self):
        """
        tbl: table look-up
        """
        self.S_idx_base = np.power(self.N_Symbols,range(self.N_A))
        All_S = np.power(self.N_Symbols,self.N_A)

        # Model: All states x all actions x all associated new states x (reward, prob)
        self.N_Model = np.zeros((All_S, self.N_A), dtype=np.int) #(Reward, p)
        self.Model_R_N = np.zeros((All_S, self.N_A), dtype=np.float32) #(Reward, p)
        self.Model_Done1_N = np.zeros((All_S, self.N_A), dtype=np.float32) #(Reward, p)
        self.Model_P_N = np.zeros((All_S, self.N_A, self.N_A), dtype=np.float32) #(Reward, p)
        self.Model_Done2_N = np.zeros((All_S, self.N_A, self.N_A), dtype=np.float32) #(Reward, p)

    def modeling_tabl_update(self, S, action, reward, op_action, done):
        S_idx = np.sum(self.S_idx_base * S)
        self.Model_R_N[S_idx, action] += reward
        self.Model_Done1_N[S_idx, action] += 1 if done else 0
        self.N_Model[S_idx, action] += 1
        if op_action is not None:
            # Model_Done1_N is 1, then Done2_N is no need to consider. 
            self.Model_P_N[S_idx, action, op_action] += 1.0                    
            self.Model_Done2_N[S_idx, action, op_action] += 1 if done else 0

    def modeling_qlearn(self, N_episodes=2, disp_flag=False):
        """
        Modeling of the game system is performed. 

        Return
        ------
        Not implemented yet.
        """        
        S_idx_base = np.power(self.N_Symbols,range(self.N_A))
        All_S = np.power(self.N_Symbols,self.N_A)
        # Model: All states x all actions x all associated new states x (reward, prob)
        self.N_Model = np.zeros((All_S, self.N_A), dtype=np.int) #(Reward, p)
        self.Model_R_N = np.zeros((All_S, self.N_A), dtype=np.float32) #(Reward, p)
        self.Model_Done1_N = np.zeros((All_S, self.N_A), dtype=np.float32) #(Reward, p)
        self.Model_P_N = np.zeros((All_S, self.N_A, self.N_A), dtype=np.float32) #(Reward, p)
        self.Model_Done2_N = np.zeros((All_S, self.N_A, self.N_A), dtype=np.float32) #(Reward, p)
        # Opponent player index
        P_no = 1 # player Q function, regardless of play order (first or next)
        play_order = 1
        ttt_env = Tictactoe_Env(self.N_A, play_order=play_order) #both X but start 1st and 2nd
        for _ in range(N_episodes):
            S, _ = ttt_env.reset(play_order=play_order)
            done = False            
            while not done:
                action, _ = self.get_action(P_no, S)
                S_new, _, reward, done, op_action = ttt_env.step_op(action)
                S_idx = np.sum(S_idx_base * S)

                self.Model_R_N[S_idx, action] += reward
                self.Model_Done1_N[S_idx, action] += 1 if done else 0
                self.N_Model[S_idx, action] += 1
                if op_action is not None:
                    # Model_Done1_N is 1, then Done2_N is no need to consider. 
                    self.Model_P_N[S_idx, action, op_action] += 1.0                    
                    self.Model_Done2_N[S_idx, action, op_action] += 1 if done else 0

                if disp_flag:
                    print(f'S_idx, action, op_action = {S_idx}, {action}, {op_action}')
                    print(f'Model_R, N_Model =',
                            self.Model_R_N[S_idx, action],
                            self.N_Model[S_idx, action])
                    if op_action is not None:
                        print(f'Model_P =',
                                self.Model_P_N[S_idx, action, op_action])

                S = S_new
            play_order = 1 if play_order == 1 else 2

    def _learning_qlearn_variable_epsilon(self, N_episodes=2, ff=0.9, lr=0.01, epsilon_d = 0.4, print_cnt=10):
        """Return: 
            cnt_trace = [cnt, ...]: cnt vector are stacked in cnt_trace
        """
        if np.isscalar(epsilon_d):
            self.learning_qlearn(N_episodes=N_episodes, ff=ff, lr=lr, epsilon = epsilon_d, print_cnt=print_cnt)

        cnt = [0, 0, 0, 0, 0, 0, 0, 0, 0] # tie, p1, p2
        cnt_trace = [cnt.copy()]        

        # Opponent player index
        P_no = 1 # player Q function, regardless of play order (first or next)
        play_order = 1
        ttt_env = Tictactoe_Env(self.N_A, play_order=play_order) #both X but start 1st and 2nd
        self.epsilon = epsilon_d['first value']
        for episode in range(N_episodes):
            if episode == epsilon_d['epsilon change episode']:                           
                self.epsilon = epsilon_d['second value']

            S, _ = ttt_env.reset(play_order=play_order)
            done = False            
            Replay_buff_d = {'S':[], 'action': [], 'S_new': [], 'reward': []}
            while not done:
                action, _ = self.get_action(P_no, S)
                S_new, _, reward, done = ttt_env.step(action)

                # Save to replay buffers
                Replay_buff_d['S'].append(S.copy())
                Replay_buff_d['action'].append(action)
                Replay_buff_d['S_new'].append(S_new.copy())
                Replay_buff_d['reward'].append(reward)
                
                # print(episode, [S, action, S_new, reward])
                S = S_new

            #######################################
            # DQN start, here for learning   
            #######################################
            # print('play_order, P_no = ', play_order, P_no)

            if Replay_buff_d['reward'][-1] == 1.0: 
                cnt[1] += 1               # P_no = 1 
                cnt[2 + play_order] += 1  # player_order = 1
                cnt[4 + play_order] += 1  # player_order | P_no = 1, so it occupied 2 lists
            elif Replay_buff_d['reward'][-1] == 0.5:
                cnt[0] += 1 
            else: # play_order = 2
                cnt[2] += 1                   # P_no = 2  
                cnt[2 + 3 - play_order] += 1  # player_order = 2
                cnt[6 + 3 - play_order] += 1  # player_order | P_no = 2, so it start from 6

            cnt_trace.append(cnt.copy())

            if episode % print_cnt == 0:
                print(episode, cnt)                
                print('S = [0,0,0, 0,0,0, 0,0,0]')
                print('Qsa[0][0,:]', [f'{self.Qsa[0][0,a]:.1e}' for a in range(9)])
                print('Qsa[1][0,:]', [f'{self.Qsa[1][0,a]:.1e}' for a in range(9)])
                print('Exproration: Epsilon=', self.epsilon)

            # Q-learning
            random_shuffle_dict_inplace(Replay_buff_d)
            for j in range(len(Replay_buff_d['reward'])):
                S = Replay_buff_d['S'][j]
                action = Replay_buff_d['action'][j]
                S_new = Replay_buff_d['S_new'][j]
                reward = Replay_buff_d['reward'][j]
                S_new_idx = calc_S_idx_numba(S_new, self.N_Symbols)
                y = reward + ff * np.max(self.Qsa[P_no-1][S_new_idx,:])
                S_idx = calc_S_idx_numba(S, self.N_Symbols)
                self.Qsa[P_no-1][S_idx, action] += lr * y

            play_order = 3 - play_order # 1 --> 2, 2 --> 1

        return cnt_trace                        

    def learning_qlearn_variable_epsilon(self, N_episodes=2, ff=0.9, lr=0.01, epsilon_d = 0.4, print_cnt=10):
        """Return: 
            cnt_trace = [cnt, ...]: cnt vector are stacked in cnt_trace
        """
        if np.isscalar(epsilon_d):
            self.learning_qlearn(N_episodes=N_episodes, ff=ff, lr=lr, epsilon = epsilon_d, print_cnt=print_cnt)

        #cnt = [0, 0, 0, 0, 0, 0, 0, 0, 0] # tie, p1, p2
        #cnt_trace = [cnt.copy()]        
        cnt_trace = CntTrace()

        # Opponent player index
        P_no = 1 # player Q function, regardless of play order (first or next)
        play_order = 1
        ttt_env = Tictactoe_Env(self.N_A, play_order=play_order) #both X but start 1st and 2nd
        self.epsilon = epsilon_d['first value']
        for episode in range(N_episodes):
            if episode == epsilon_d['epsilon change episode']:                           
                self.epsilon = epsilon_d['second value']

            S, _ = ttt_env.reset(play_order=play_order)
            done = False            
            aReplay_buff = ReplayBuff()
            # Expereince to collect data
            while not done:
                action, _ = self.get_action(P_no, S)
                S_new, _, reward, done = ttt_env.step(action)
                aReplay_buff.append(S.copy(), action, S_new.copy(), reward, done)
                S = S_new
            # cnt_trace.calc_and_append(aReplay_buff.d['reward'][-1], play_order)   
            cnt_trace.calc_and_append(reward, play_order)
            # Q-learning
            self.q_learning_batch(aReplay_buff.d, P_no=P_no, ff=ff, lr=lr)

            if episode % print_cnt == 0:
                self.disp_Qsa_state0(episode, cnt_trace.cnt)


            play_order = 3 - play_order # 1 --> 2, 2 --> 1

        return cnt_trace.buff   

    def learning_planning_qlearn_variable_epsilon(self, N_episodes=2, N_plan=0, ff=0.9, lr=0.01, epsilon_d = None, print_cnt=10):
        """Return: 
            cnt_trace = [cnt, ...]: cnt vector are stacked in cnt_trace
        """
        env_model = EnvModel_TBL(self.N_A, self.N_Symbols)

        cnt_trace = CntTrace()

        # Opponent player index
        P_no = 1 # player Q function, regardless of play order (first or next)
        play_order = 1
        ttt_env = Tictactoe_Env(self.N_A, play_order=play_order) #both X but start 1st and 2nd
        self.epsilon = epsilon_d['first value']
        for episode in range(N_episodes):
            if episode == epsilon_d['epsilon change episode']:                           
                self.epsilon = epsilon_d['second value']

            S, _ = ttt_env.reset(play_order=play_order)
            done = False            
            aReplay_buff = ReplayBuff()
            # Expereince to collect data
            while not done:
                action, _ = self.get_action(P_no, S)
                S_new, _, reward, done, op_action = ttt_env.step_op(action)
                aReplay_buff.append(S.copy(), action, S_new.copy(), reward, done)
                env_model.update(S, action, reward, op_action, done)
                S = S_new
            # cnt_trace.calc_and_append(aReplay_buff.d['reward'][-1], play_order)
            cnt_trace.calc_and_append(reward, play_order)
            # Q-learning
            self.q_learning_batch(aReplay_buff.d, P_no=P_no, ff=ff, lr=lr)
            self.q_learning_planning_batch(env_model, aReplay_buff.d, N_plan=N_plan, P_no=P_no, ff=ff, lr=lr)

            if episode % print_cnt == 0:
                self.disp_Qsa_state0(episode, cnt_trace.cnt)
                print('Display modeling status')
                S = [1, 1, 0, 2, 2, 0, 0, 0, 0]
                env_model.disp(S, action=2)
                print()

            play_order = 3 - play_order # 1 --> 2, 2 --> 1

        return cnt_trace.buff       

    def disp_Qsa_state0(self, episode, cnt):
        print('-------------------------')
        print(episode, cnt)                
        print('S = [0,0,0, 0,0,0, 0,0,0]')
        print('Qsa[0][0,:]', [f'{self.Qsa[0][0,a]:.1e}' for a in range(9)])
        print('Qsa[1][0,:]', [f'{self.Qsa[1][0,a]:.1e}' for a in range(9)])
        print('Exproration: Epsilon=', self.epsilon)
        print()

    def playing_random(self, N_episodes=2, print_cnt=10):
        """Plyaing two random players. To make baseline performance so that it will compared with learnt agents.
        - Check whethere it requires ff, lr, which may not be useful in this case since no learning is applied. 
        ------
        Return 
            cnt_trace = [cnt, ...]: cnt vector are stacked in cnt_trace
        """
        cnt = [0, 0, 0, 0, 0] # tie, p1, p2
        cnt_trace = [cnt.copy()]        

        # Opponent player index
        # P_no = 1 # player Q function, regardless of play order (first or next)
        play_order = 1
        ttt_env = Tictactoe_Env(self.N_A, play_order=play_order) #both X but start 1st and 2nd
        for episode in range(N_episodes):
            S, _ = ttt_env.reset(play_order=play_order)
            done = False            
            Replay_buff = []
            while not done:
                # self.epsilon = epsilon # epsilon is a hyperparamter for exploration
                action = ttt_env.sample_action()
                S_new, _, reward, done = ttt_env.step(action)
                Replay_buff.append([S.copy(), action, S_new.copy(), reward])
                # print(episode, [S, action, S_new, reward])
                S = S_new

            #######################################
            # DQN start, here for learning   
            #######################################
            # print('play_order, P_no = ', play_order, P_no)

            if Replay_buff[-1][3] == 1.0: 
                cnt[1] += 1               # play_order = 1 
                cnt[2 + play_order] += 1  # P_no = 1 (first player)
            elif Replay_buff[-1][3] == 0.5:
                cnt[0] += 1   
                # cnt[2 + 3 - play_order] += 1  # P_no = 2 (second player)
            else: # play_order = 2
                cnt[2] += 1
                cnt[2 + 3 - play_order] += 1

            cnt_trace.append(cnt.copy())

            if episode % print_cnt == 0:
                print(episode, cnt)                
                print('S = [0,0,0, 0,0,0, 0,0,0]')
                print('Qsa[0][0,:]', [f'{self.Qsa[0][0,a]:.1e}' for a in range(9)])
                print('Qsa[1][0,:]', [f'{self.Qsa[1][0,a]:.1e}' for a in range(9)])
                print('Exproration: Epsilon=', self.epsilon)

            play_order = 3 - play_order # 1 --> 2, 2 --> 1

        return cnt_trace


def learning_stage_qlearn(N_episodes=100, epsilon=0.4, save_flag=True, fig_flag=False):
    ff = 0.9
    lr = 0.01
    N_Symbols = 3 # 0=empty, 1=plyaer1, 2=player2
    N_A = 9 # (0,0), (0,1), ..., (2,2)
    print_cnt = N_episodes / 10

    my_Q_System = Q_System_QL(N_A, N_Symbols)
    cnt_trace = my_Q_System.learning_qlearn(N_episodes=N_episodes, ff=ff, lr=lr, epsilon=epsilon, print_cnt=print_cnt)

    print('-------------------')
    cnt_last = cnt_trace[-1]
    cnt_last_normal = np.array(cnt_last) / np.sum(cnt_last[0:3])
    # Showing normalized counts as well so as to make feel the progress.
    # In random agent playing here, no progress should be displayed.
    print(N_episodes, f"Last cnt:{cnt_last}, Normalized last cnt:{cnt_last_normal}")

    if save_flag:
        my_Q_System.save()

    if fig_flag:
        plot_cnt_trace_normal_order_detail(cnt_trace, title='Q-learning')

    return my_Q_System

def learning_planning_stage_qlearn_variable_epsilon(N_episodes=100, epsilon_d=0.4, N_plan=0, save_flag=True, fig_flag=False):
    ff = 0.9
    lr = 0.01
    N_Symbols = 3 # 0=empty, 1=plyaer1, 2=player2
    N_A = 9 # (0,0), (0,1), ..., (2,2)
    print_cnt = N_episodes / 10

    my_Q_System = Q_System_QL(N_A, N_Symbols)
    cnt_trace = my_Q_System.learning_planning_qlearn_variable_epsilon(N_episodes=N_episodes, N_plan=N_plan, ff=ff, lr=lr, epsilon_d=epsilon_d, print_cnt=print_cnt)

    print('-------------------')
    cnt_last = cnt_trace[-1]
    cnt_last_normal = np.array(cnt_last) / np.sum(cnt_last[0:3])
    # Showing normalized counts as well so as to make feel the progress.
    # In random agent playing here, no progress should be displayed.
    print(N_episodes, f"Last cnt:{cnt_last}, Normalized last cnt:{cnt_last_normal}")

    if save_flag:
        my_Q_System.save()

    if fig_flag:
        plot_cnt_trace_normal_order_detail(cnt_trace, title='Q-learning')

    return my_Q_System

def learning_stage_qlearn_variable_epsilon(N_episodes=100, epsilon_d=0.4, save_flag=True, fig_flag=False):
    if np.isscalar(epsilon_d):
        return learning_stage_qlearn(N_episodes=N_episodes, epsilon=epsilon_d, save_flag=save_flag, fig_flag=fig_flag)
    # assert epsilon_d is None, 'epsilon_d should be dictionary type.'

    ff = 0.9
    lr = 0.01
    N_Symbols = 3 # 0=empty, 1=plyaer1, 2=player2
    N_A = 9 # (0,0), (0,1), ..., (2,2)
    print_cnt = N_episodes / 10

    my_Q_System = Q_System_QL(N_A, N_Symbols)
    cnt_trace = my_Q_System.learning_qlearn_variable_epsilon(N_episodes=N_episodes, ff=ff, lr=lr, epsilon_d=epsilon_d, print_cnt=print_cnt)

    print('-------------------')
    cnt_last = cnt_trace[-1]
    cnt_last_normal = np.array(cnt_last) / np.sum(cnt_last[0:3])
    # Showing normalized counts as well so as to make feel the progress.
    # In random agent playing here, no progress should be displayed.
    print(N_episodes, f"Last cnt:{cnt_last}, Normalized last cnt:{cnt_last_normal}")

    if save_flag:
        my_Q_System.save()

    if fig_flag:
        plot_cnt_trace_normal_order_detail(cnt_trace, title='Q-learning')

    return my_Q_System


def learning_stage_dqn(N_episodes=100, epsilon=0.2, save_flag=True, fig_flag=False):
    ff = 0.9
    lr = 0.01
    N_Symbols = 3 # 0=empty, 1=plyaer1, 2=player2
    N_A = 9 # (0,0), (0,1), ..., (2,2)
    print_cnt = N_episodes / 10

    my_Q_System = Q_System_DQN(N_A, N_Symbols)
    cnt_trace = my_Q_System.learning_dqn(N_episodes=N_episodes, ff=ff, lr=lr, epsilon=epsilon, print_cnt=print_cnt)

    print('-------------------')
    cnt_last = cnt_trace[-1]
    cnt_last_normal = np.array(cnt_last) / np.sum(cnt_last[0:3])
    # Showing normalized counts as well so as to make feel the progress.
    # In random agent playing here, no progress should be displayed.
    print(N_episodes, f"Last cnt:{cnt_last}, Normalized last cnt:{cnt_last_normal}")

    if save_flag:
        my_Q_System.save()

    if fig_flag:
        plot_cnt_trace_normal_order_detail(cnt_trace, title='Q-learning')

    return my_Q_System    

def playing_stage_random(N_episodes=100, fig_flag=False):
    N_Symbols = 3 # 0=empty, 1=plyaer1, 2=player2
    N_A = 9 # (0,0), (0,1), ..., (2,2)
    print_cnt = N_episodes / 10

    my_Q_System = Q_System_QL(N_A, N_Symbols)
    #cnt_trace = my_Q_System.learning(N_episodes=N_episodes, ff=ff, lr=lr, print_cnt=print_cnt)
    cnt_trace = my_Q_System.playing_random(N_episodes=N_episodes, print_cnt=print_cnt)

    print('-------------------')
    cnt_last = cnt_trace[-1]
    cnt_last_normal = np.array(cnt_last) / np.sum(cnt_last[0:3])
    # Showing normalized counts as well so as to make feel the progress.
    # In random agent playing here, no progress should be displayed.
    print(N_episodes, f"Last cnt:{cnt_last}, Normalized last cnt:{cnt_last_normal}")

    if fig_flag:
        plot_cnt_trace_normal_order(cnt_trace, title='Playing by random agents')

    return my_Q_System


def playing_stage_dqn(N_episodes=100, fig_flag=False):
    N_Symbols = 3 # 0=empty, 1=plyaer1, 2=player2
    N_A = 9 # (0,0), (0,1), ..., (2,2)
    print_cnt = N_episodes / 10

    my_Q_System = Q_System_DQN(N_A, N_Symbols)
    #cnt_trace = my_Q_System.learning(N_episodes=N_episodes, ff=ff, lr=lr, print_cnt=print_cnt)
    cnt_trace = my_Q_System.playing_dqn(N_episodes=N_episodes, print_cnt=print_cnt)

    print('-------------------')
    cnt_last = cnt_trace[-1]
    cnt_last_normal = np.array(cnt_last) / np.sum(cnt_last[0:3])
    # Showing normalized counts as well so as to make feel the progress.
    # In random agent playing here, no progress should be displayed.
    print(N_episodes, f"Last cnt:{cnt_last}, Normalized last cnt:{cnt_last_normal}")

    if fig_flag:
        plot_cnt_trace_normal_order(cnt_trace, title='Playing by random agents')


def _r1_main():
    """
    We will use Deep Q-Networks for learning. 
    """
    Q1 = input_default('1. Loading a trained agent (0) or Learning a new agent (1)? (default=0) ', 0, int)
    if Q1 == 0:
        print('Loading the trained agent...')
        Q2 = input_default('2. Do you want to start first?(0=yes,1=no,default=0) ', 0, int)
        player_human = Q2 + 1
        if player_human == 1:
            print('You=1(X), Agent=2(O)') 
        else:
            print('Agent=1(X), You=2(O)') 
        trained_Q_System = Q_System(None)
        trained_Q_System.load()
        trained_Q_System.play_with_human_inference(player_human)
        # print(len(trained_Q_System.Qsa))
    else:
        print('Start to learn a new agent...')
        Q2 = input_default('2. How many episode do you want to learn?(default=10000) ', 10000, int)
        # my_Q_System = learning_stage(N_episodes=Q2, fig_flag=True)
        _ = learning_stage_qlearn(N_episodes=Q2, fig_flag=True)
        # print(len(my_Q_System.Qsa))


class MLP_AGENT(tf.keras.layers.Layer):
    """MLP Model for our agent"""
    def __init__(self, S_ln=9, action_ln=9, hidden_ln=9):
        super(MLP_AGENT, self).__init__()
        self.linear_1 = tf.keras.layers.Dense(S_ln + action_ln, activation=tf.nn.relu)
        self.linear_2 = tf.keras.layers.Dense(hidden_ln, activation=tf.nn.relu)
        self.linear_3 = tf.keras.layers.Dense(1) # no activation

    def call(self, inputs):
        x = self.linear_1(inputs)
        x = self.linear_2(x)
        return self.linear_3(x)

class Q_System_DQN(Q_System_QL):
    def __init__(self, N_A=9, N_Symbols=3):
        super(Q_System_DQN, self).__init__(N_A=N_A, N_Symbols=N_Symbols)
        if N_A is not None:
            self.QSA_net = [MLP_AGENT(self.N_A, self.N_A, self.N_A), MLP_AGENT(self.N_A, self.N_A, self.N_A)]

    def save_id(self, file_name='tictactoe_data_dqn.pckl', class_name='Q_System_DQN'):
        '''This is common for DQN and CNNDQN'''
        f = open(file_name, 'wb')
        W_p1 = [W.numpy() for W in self.QSA_net[0].weights]
        W_p2 = [W.numpy() for W in self.QSA_net[1].weights]
        obj = [self.N_A, self.N_Symbols, self.epsilon, [W_p1, W_p2], class_name]
        pickle.dump(obj, f)
        f.close()

    def save(self):
        self.save_id(file_name='tictactoe_data_dqn.pckl', class_name='Q_System_DQN')

    def load_id(self, init_in_shape, file_name='tictactoe_data_dqn.pckl'):
        f = open(file_name, 'rb')
        obj = pickle.load(f)
        [self.N_A, self.N_Symbols, self.epsilon, W_list, self.class_name] = obj
        f.close()
        _ = self.QSA_net[0](np.zeros(init_in_shape, dtype='float32'))
        self.QSA_net[0].set_weights(W_list[0])
        _ = self.QSA_net[1](np.zeros(init_in_shape, dtype='float32'))
        self.QSA_net[1].set_weights(W_list[1])

    def load(self):
        init_in_shape = (1,self.N_A*2) # S and action
        self.load_id(init_in_shape, file_name='tictactoe_data_dqn.pckl')

    def _get_q_net(self, S, action_list, P_no):
        action_prob = []
        # S_idx = self.calc_S_idx(S)
        #for a in action_list:
        #    action_prob.append(self.Qsa[P_no-1][S_idx,a])  

        S_in = np.array(S,dtype=float).reshape(1,-1)
        QSA_all_actions = self.QSA_net[P_no-1](S_in).numpy()[0]
        for a in action_list:
            action_prob.append(QSA_all_actions[a])  

        return action_prob

    def get_q_net(self, S:np.ndarray, action_list, P_no):
        action_prob = []
        X_in_stack = np.zeros((len(action_list), self.N_A + self.N_A))
        for i in range(len(action_list)):
            a = action_list[i]
            a_buff = [0] * self.N_A
            a_buff[a] = 1
            X_in = np.array(list(S) + a_buff, dtype=np.float32).reshape(1,-1)
            X_in_stack[i, :] = X_in[0, :]

        Qsa = self.QSA_net[P_no-1](X_in_stack).numpy()[:,0]
        action_prob = list(Qsa)
        return action_prob

    def policy(self, P_no, S, action_list):
        """Return action regardless P_no but just specify Q[P_no]
        """       
        #for a in action_list:
        #    action_prob.append(self.Qsa[P_no-1][S_idx,a])                
        action_prob = self.get_q_net(S, action_list, P_no)

        # We consider max Q with epsilon greedy
        if tf.squeeze(tf.random.uniform([1,1])) > self.epsilon:
            action = action_list[one_of_amax(action_prob)]            
        else:
            action = action_list[np.random.randint(0,len(action_list),1)[0]]
            
        if self.disp_flag:
            print('action', action, 
                  'action_list', action_list, 'action_prob', action_prob)
        return action

    def get_action(self, P_no, S):
        """
        Return action, done
        """
        action_list, no_occupied = self.find_action_list(S)
        # Since number of possible actions are reduced, 
        # denominator is also updated. 
        action = self.policy(P_no, S, action_list)
        done = no_occupied == (self.N_A - 1)
        return action, done

    def playing_dqn(self, N_episodes=2, print_cnt=10):
        """Plyaing two random players. To make baseline performance so that it will compared with learnt agents.
        - Check whethere it requires ff, lr, which may not be useful in this case since no learning is applied. 
        ------
        Return 
            cnt_trace = [cnt, ...]: cnt vector are stacked in cnt_trace
        """
        cnt = [0, 0, 0, 0, 0] # tie, p1, p2
        cnt_trace = [cnt.copy()]        

        # Opponent player index
        P_no = 1 # player Q function, regardless of play order (first or next)
        play_order = 1
        ttt_env = Tictactoe_Env(self.N_A, play_order=play_order) #both X but start 1st and 2nd
        for episode in range(N_episodes):
            S, _ = ttt_env.reset(play_order=play_order)
            done = False            
            Replay_buff = []
            while not done:
                # self.epsilon = epsilon # epsilon is a hyperparamter for exploration
                action, _ = self.get_action(P_no, list(S))
                S_new, _, reward, done = ttt_env.step(action)
                Replay_buff.append([S.copy(), action, S_new.copy(), reward])
                # print(episode, [S, action, S_new, reward])
                S = S_new

            #######################################
            # DQN start, here for learning   
            #######################################
            # print('play_order, P_no = ', play_order, P_no)

            if Replay_buff[-1][3] == 1.0: 
                cnt[1] += 1               # play_order = 1 
                cnt[2 + play_order] += 1  # P_no = 1 (first player)
            elif Replay_buff[-1][3] == 0.5:
                cnt[0] += 1   
                # cnt[2 + 3 - play_order] += 1  # P_no = 2 (second player)
            else: # play_order = 2
                cnt[2] += 1
                cnt[2 + 3 - play_order] += 1

            cnt_trace.append(cnt.copy())

            if episode % print_cnt == 0:
                print(episode, cnt)                
                print('S = [0,0,0, 0,0,0, 0,0,0]')
                S = [0] * self.N_A
                Qsa_0_0, Qsa_1_0 = [], []
                for action in range(self.N_A):
                    action_buff = [0] * self.N_A
                    action_buff[action] = 1
                    X_in = np.array(S + action_buff).reshape(1,-1)
                    Qsa_0_0.append(self.QSA_net[0](X_in).numpy()[0,0])
                    Qsa_1_0.append(self.QSA_net[1](X_in).numpy()[0,0])
                print('Qsa[0][0,:]', [f'{Qsa_0_0[a]:.1e}' for a in range(9)])
                print('Qsa[1][0,:]', [f'{Qsa_1_0[a]:.1e}' for a in range(9)])
                print('Exproration: Epsilon=', self.epsilon)

            play_order = 3 - play_order # 1 --> 2, 2 --> 1

        return cnt_trace

    def learning_dqn(self, N_episodes=2, ff=0.9, lr=0.01, epsilon=0.1, print_cnt=10):
        """Plyaing two random players. To make baseline performance so that it will compared with learnt agents.
        - Check whethere it requires ff, lr, which may not be useful in this case since no learning is applied. 
        ------
        Return 
            cnt_trace = [cnt, ...]: cnt vector are stacked in cnt_trace
        """
        cnt = [0, 0, 0, 0, 0, 0, 0, 0, 0] # tie, p1, p2
        cnt_trace = [cnt.copy()]        

        # Opponent player index
        P_no = 1 # player Q function, regardless of play order (first or next)
        play_order = 1
        ttt_env = Tictactoe_Env(self.N_A, play_order=play_order) #both X but start 1st and 2nd
        optimizer = tf.keras.optimizers.Adam()
        loss_f = tf.keras.losses.MeanSquaredError()   
        for episode in range(N_episodes):
            S, _ = ttt_env.reset(play_order=play_order)
            done = False            
            Replay_buff = []
            while not done:
                self.epsilon = epsilon # epsilon is a hyperparamter for exploration
                # self.epsilon = epsilon # epsilon is a hyperparamter for exploration
                action, _ = self.get_action(P_no, S)
                S_new, _, reward, done = ttt_env.step(action)
                Replay_buff.append([S.copy(), action, S_new.copy(), reward, done])
                # print(episode, [S, action, S_new, reward])
                S = S_new

            #######################################
            # DQN start, here for learning   
            #######################################
            # print('play_order, P_no = ', play_order, P_no)
            # No buffer shuffling
            y_list = []
            X_list = []
            for buff in Replay_buff:
                S, action, S_new, reward, done = buff
                if done:
                    y = reward
                else:
                    action_list_new, _ = self.find_action_list(S_new)
                    action_prob_array_new = self.get_q_net(S_new, action_list_new, P_no)
                    y = reward + ff*np.max(action_prob_array_new)
                y_list.append(y)
                action_buff = [0] * self.N_A
                action_buff[action] = 1
                X_list.append(list(S) + action_buff)
            y_N = np.array(y_list)
            X_Nx18 = np.array(X_list)
            with tf.GradientTape() as tape:
                Qsa_N = self.QSA_net[P_no-1](X_Nx18)
                loss_value = loss_f(y_N, Qsa_N)
            gradients = tape.gradient(loss_value, self.QSA_net[P_no-1].trainable_weights)
            optimizer.apply_gradients(zip(gradients, self.QSA_net[P_no-1].trainable_weights))

            if Replay_buff[-1][3] == 1.0: 
                cnt[1] += 1               # P_no = 1 
                cnt[2 + play_order] += 1  # player_order = 1
                cnt[4 + play_order] += 1  # player_order | P_no = 1, so it occupied 2 lists
            elif Replay_buff[-1][3] == 0.5:
                cnt[0] += 1 
            else: # play_order = 2
                cnt[2] += 1                   # P_no = 2  
                cnt[2 + 3 - play_order] += 1  # player_order = 2
                cnt[6 + 3 - play_order] += 1  # player_order | P_no = 2, so it start from 6

            cnt_trace.append(cnt.copy())

            if episode % print_cnt == 0:
                print()
                print('Finished episode: #', episode)
                print('cnt: ', cnt)
                print(f'Play order:{play_order}, P_no:{P_no}')
                print('S = [0,0,0, 0,0,0, 0,0,0]')
                S = [0] * self.N_A
                Qsa_0_0, Qsa_1_0 = [], []
                for action in range(self.N_A):
                    action_buff = [0] * self.N_A
                    action_buff[action] = 1
                    X_in = np.array(S + action_buff).reshape(1,-1)
                    Qsa_0_0.append(self.QSA_net[0](X_in).numpy()[0,0])
                    Qsa_1_0.append(self.QSA_net[1](X_in).numpy()[0,0])
                print('Qsa[0][0,:]', [f'{Qsa_0_0[a]:.1e}' for a in range(9)])
                print('Qsa[1][0,:]', [f'{Qsa_1_0[a]:.1e}' for a in range(9)])
                print('Exproration: Epsilon=', self.epsilon)

            play_order = 3 - play_order # 1 --> 2, 2 --> 1

        return cnt_trace
 
    def learning_dqn_variable_epsilon(self, N_episodes=2, 
            ff=0.9, lr=0.01, epsilon_d=0.1, print_cnt=10,
            agent_type:int=0):
        """Plyaing two random players. To make baseline performance so that it will compared with learnt agents.
        - Check whethere it requires ff, lr, which may not be useful in this case since no learning is applied. 
        ------
        Return 
            cnt_trace = [cnt, ...]: cnt vector are stacked in cnt_trace
        """
        if np.isscalar(epsilon_d):
            return self.learning_dqn(N_episodes=N_episodes, ff=ff, lr=lr, epsilon=epsilon_d, print_cnt=print_cnt)

        cnt = [0, 0, 0, 0, 0, 0, 0, 0, 0] # tie, p1, p2
        cnt_trace = [cnt.copy()]        

        # Opponent player index
        P_no = 1 # player Q function, regardless of play order (first or next)
        play_order = 1
        # ttt_env = Tictactoe_Env(self.N_A, play_order=play_order) #both X but start 1st and 2nd
        ttt_env = Tictactoe_Env_Active(self.N_A, play_order=play_order, 
                agent_type=agent_type, Q_System_Class=Q_System_DQN)
        optimizer = tf.keras.optimizers.Adam()
        loss_f = tf.keras.losses.MeanSquaredError()   

        self.epsilon = epsilon_d['first value']
        for episode in range(N_episodes):
            if episode == epsilon_d['epsilon change episode']:                           
                self.epsilon = epsilon_d['second value']

            S, _ = ttt_env.reset(play_order=play_order)
            done = False            
            Replay_buff = []
            while not done:
                # self.epsilon = epsilon # epsilon is a hyperparamter for exploration
                action, _ = self.get_action(P_no, S)
                S_new, _, reward, done = ttt_env.step(action)
                Replay_buff.append([S.copy(), action, S_new.copy(), reward, done])
                # print(episode, [S, action, S_new, reward])
                S = S_new

            #######################################
            # DQN start, here for learning   
            #######################################
            # print('play_order, P_no = ', play_order, P_no)
            # No buffer shuffling
            y_list = []
            X_list = []
            for buff in Replay_buff:
                S, action, S_new, reward, done = buff
                if done:
                    y = reward
                else:
                    action_list_new, _ = self.find_action_list(S_new)
                    action_prob_array_new = self.get_q_net(S_new, action_list_new, P_no)
                    y = reward + ff*np.max(action_prob_array_new)
                y_list.append(y)
                action_buff = [0] * self.N_A
                action_buff[action] = 1
                X_list.append(list(S) + action_buff)
            y_N = np.array(y_list)
            X_Nx18 = np.array(X_list)
            with tf.GradientTape() as tape:
                Qsa_N = self.QSA_net[P_no-1](X_Nx18)
                loss_value = loss_f(y_N, Qsa_N)
            gradients = tape.gradient(loss_value, self.QSA_net[P_no-1].trainable_weights)
            optimizer.apply_gradients(zip(gradients, self.QSA_net[P_no-1].trainable_weights))

            if Replay_buff[-1][3] == 1.0: 
                cnt[1] += 1               # P_no = 1 
                cnt[2 + play_order] += 1  # player_order = 1
                cnt[4 + play_order] += 1  # player_order | P_no = 1, so it occupied 2 lists
            elif Replay_buff[-1][3] == 0.5:
                cnt[0] += 1 
            else: # play_order = 2
                cnt[2] += 1                   # P_no = 2  
                cnt[2 + 3 - play_order] += 1  # player_order = 2
                cnt[6 + 3 - play_order] += 1  # player_order | P_no = 2, so it start from 6

            cnt_trace.append(cnt.copy())

            if episode % print_cnt == 0:
                print()
                print('Finished episode: #', episode)
                print('cnt: ', cnt)
                print(f'Play order:{play_order}, P_no:{P_no}')
                print('S = [0,0,0, 0,0,0, 0,0,0]')
                S = [0] * self.N_A
                Qsa_0_0, Qsa_1_0 = [], []
                for action in range(self.N_A):
                    action_buff = [0] * self.N_A
                    action_buff[action] = 1
                    X_in = np.array(S + action_buff).reshape(1,-1)
                    Qsa_0_0.append(self.QSA_net[0](X_in).numpy()[0,0])
                    Qsa_1_0.append(self.QSA_net[1](X_in).numpy()[0,0])
                print('Qsa[0][0,:]', [f'{Qsa_0_0[a]:.1e}' for a in range(9)])
                print('Qsa[1][0,:]', [f'{Qsa_1_0[a]:.1e}' for a in range(9)])
                print('Exproration: Epsilon=', self.epsilon)

            play_order = 3 - play_order # 1 --> 2, 2 --> 1

        return cnt_trace

def _learning_stage_dqn_variable_epsilon(N_episodes=100, epsilon_d=0.2, save_flag=True, fig_flag=False):
    if np.isscalar(epsilon_d):
        return learning_stage_dqn(N_episodes=N_episodes, epsilon=epsilon_d, save_flag=save_flag, fig_flag=fig_flag)

    ff = 0.9
    lr = 0.01
    N_Symbols = 3 # 0=empty, 1=plyaer1, 2=player2
    N_A = 9 # (0,0), (0,1), ..., (2,2)
    print_cnt = N_episodes / 10

    my_Q_System = Q_System_DQN(N_A, N_Symbols)
    cnt_trace = my_Q_System.learning_dqn_variable_epsilon(N_episodes=N_episodes, ff=ff, lr=lr, epsilon_d=epsilon_d, print_cnt=print_cnt)

    print('-------------------')
    cnt_last = cnt_trace[-1]
    cnt_last_normal = np.array(cnt_last) / np.sum(cnt_last[0:3])
    # Showing normalized counts as well so as to make feel the progress.
    # In random agent playing here, no progress should be displayed.
    print(N_episodes, f"Last cnt:{cnt_last}, Normalized last cnt:{cnt_last_normal}")

    if save_flag:
        my_Q_System.save()

    if fig_flag:
        plot_cnt_trace_normal_order_detail(cnt_trace, title='Q-learning')

    return my_Q_System
 
def learning_stage_dqn_variable_epsilon(N_episodes=100, epsilon_d=0.2, 
        agent_type:int=0, save_flag=True, fig_flag=False):
    if np.isscalar(epsilon_d):
        return learning_stage_dqn(N_episodes=N_episodes, epsilon=epsilon_d, save_flag=save_flag, fig_flag=fig_flag)

    ff = 0.9
    lr = 0.01
    N_Symbols = 3 # 0=empty, 1=plyaer1, 2=player2
    N_A = 9 # (0,0), (0,1), ..., (2,2)
    print_cnt = N_episodes / 10

    my_Q_System = Q_System_DQN(N_A, N_Symbols)
    cnt_trace = my_Q_System.learning_dqn_variable_epsilon(N_episodes=N_episodes, 
                    ff=ff, lr=lr, epsilon_d=epsilon_d, print_cnt=print_cnt,
                    agent_type=agent_type)

    print('-------------------')
    cnt_last = cnt_trace[-1]
    cnt_last_normal = np.array(cnt_last) / np.sum(cnt_last[0:3])
    # Showing normalized counts as well so as to make feel the progress.
    # In random agent playing here, no progress should be displayed.
    print(N_episodes, f"Last cnt:{cnt_last}, Normalized last cnt:{cnt_last_normal}")

    if save_flag:
        my_Q_System.save()

    if fig_flag:
        plot_cnt_trace_normal_order_detail(cnt_trace, title='Q-learning')

    return my_Q_System

class CNN_AGENT(tf.keras.layers.Layer):
    """CNN Model for our agent"""
    def __init__(self, S_ln=9, action_ln=9, hidden_ln=9):
        super(CNN_AGENT, self).__init__()
        self.cnn_2d_1 = tf.keras.layers.Conv2D(hidden_ln, (2,2), padding='same', activation=tf.nn.relu)
        self.cnn_2d_2 = tf.keras.layers.Conv2D(hidden_ln, (2,2), padding='same', activation=tf.nn.relu)
        self.linear_1 = tf.keras.layers.Dense(1) # no activation

    def call(self, inputs_2d2c):
        chanDim = -1
        x = self.cnn_2d_1(inputs_2d2c)
        #x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))(x)
        x = tf.keras.layers.BatchNormalization(axis=chanDim)(x)
        x = self.cnn_2d_2(x)
        #x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))(x)
        x = tf.keras.layers.BatchNormalization(axis=chanDim)(x)
        x = tf.keras.layers.Flatten()(x)
        return self.linear_1(x)


def get_X_in_stack(N_A:int, sqrt_n_a:int, action_list_array:np.ndarray, S_array_2d:np.ndarray):
    X_in_stack = np.zeros((len(action_list_array), 2, sqrt_n_a, sqrt_n_a), dtype='float32')
    for i in range(action_list_array.shape[0]):
        a = action_list_array[i]
        X_in_stack[i, 0] = S_array_2d
        X_in_stack[i, 1, a//sqrt_n_a, a%sqrt_n_a] = 1
    return X_in_stack

@jit
def get_X_in_stack_numba(N_A:int, sqrt_n_a:int, action_list_array:np.ndarray, S_array_2d:np.ndarray):
    X_in_stack = np.zeros((len(action_list_array), 2, sqrt_n_a, sqrt_n_a), dtype='float32')
    for i in range(action_list_array.shape[0]):
        a = action_list_array[i]
        X_in_stack[i, 0] = S_array_2d
        X_in_stack[i, 1, a//sqrt_n_a, a%sqrt_n_a] = 1
    return X_in_stack

def cnn_learning_inline(QSA_net_each, X_N, y_N, loss_f, optimizer):
    with tf.GradientTape() as tape:
        Qsa_N = QSA_net_each(X_N)
        loss_value = loss_f(y_N, Qsa_N)
    gradients = tape.gradient(loss_value, QSA_net_each.trainable_weights)
    optimizer.apply_gradients(zip(gradients, QSA_net_each.trainable_weights))       

class Q_System_CNNDQN(Q_System_DQN):
    def __init__(self, N_A=9, N_Symbols=3):
        super(Q_System_CNNDQN, self).__init__(N_A=N_A, N_Symbols=N_Symbols)
        if N_A is not None:
            self.QSA_net = [CNN_AGENT(self.N_A, self.N_A, self.N_A), CNN_AGENT(self.N_A, self.N_A, self.N_A)]
            self.sqrt_n_a = int(np.sqrt(self.N_A))

    def _save(self):
        QSA_net_file_list = ['QSA_net_p1', 'QSA_net_p2']

        f = open('tictactoe_data.pckl', 'wb')
        obj = [self.N_A, self.N_Symbols, self.epsilon, QSA_net_file_list, 'Q_System_CNNDQN']
        pickle.dump(obj, f)
        f.close()

        for i, fname in enumerate(QSA_net_file_list):
            self.QSA_net[i].save_weights(fname)

    def _r1_save(self):
        # weights should be saving to array form, in order to use in load
        f = open('tictactoe_data_cnndqn.pckl', 'wb')
        W_p1 = [W.numpy() for W in self.QSA_net[0].weights]
        W_p2 = [W.numpy() for W in self.QSA_net[1].weights]
        obj = [self.N_A, self.N_Symbols, self.epsilon, [W_p1, W_p2], 'Q_System_CNNDQN']
        pickle.dump(obj, f)
        f.close()

    def save(self):
        self.save_id(file_name='tictactoe_data_cnndqn.pckl', 
                class_name='Q_System_CNNDQN')

    def _load(self):
        f = open('tictactoe_data.pckl', 'rb')
        obj = pickle.load(f)
        [self.N_A, self.N_Symbols, self.epsilon, QSA_net_file_list, self.class_name] = obj
        f.close()

        for i, fname in enumerate(QSA_net_file_list):
            self.QSA_net[i].load_weights(fname)

    def _r1_load(self):
        f = open('tictactoe_data_cnndqn.pckl', 'rb')
        obj = pickle.load(f)
        [self.N_A, self.N_Symbols, self.epsilon, W_list, self.class_name] = obj
        f.close()

        sqrt_n_a = int(np.sqrt(self.N_A))
        _ = self.QSA_net[0](np.zeros((1,2,sqrt_n_a,sqrt_n_a), dtype='float32'))
        self.QSA_net[0].set_weights(W_list[0])
        _ = self.QSA_net[1](np.zeros((1,2,sqrt_n_a,sqrt_n_a), dtype='float32'))
        self.QSA_net[1].set_weights(W_list[1])

    def load(self):
        sqrt_n_a = self.sqrt_n_a
        init_in_shape = (1,2,sqrt_n_a,sqrt_n_a)
        self.load_id(init_in_shape, file_name='tictactoe_data_cnndqn.pckl')

    def make_X_in(self, S, action_buff):
        X_in_each = np.zeros((2, self.sqrt_n_a, self.sqrt_n_a), dtype='float32')
        X_in_each[0] = np.array(S, dtype='float32').reshape(self.sqrt_n_a, self.sqrt_n_a)
        X_in_each[1] = np.array(action_buff, 'float32').reshape(self.sqrt_n_a, self.sqrt_n_a)
        return X_in_each

    def _get_q_net(self, S:np.ndarray, action_list, P_no):
        action_prob = []
        sqrt_n_a = int(np.sqrt(self.N_A))
        X_in_stack = np.zeros((len(action_list), 2, sqrt_n_a, sqrt_n_a))
        for i in range(len(action_list)):
            a = action_list[i]
            action_buff = [0] * self.N_A
            action_buff[a] = 1
            X_in_stack[i] = self.make_X_in(S, action_buff)
        Qsa = self.QSA_net[P_no-1](X_in_stack).numpy()[:,0]
        action_prob = list(Qsa)
        return action_prob

    def get_q_net(self, S:np.ndarray, action_list, P_no):
        # S_array_2d is 2-D array such as [3,3] for convolutional use
        S_array_2d = np.array(S, dtype='float32').reshape(self.sqrt_n_a, self.sqrt_n_a)
        action_list_array = np.array(action_list, dtype='int')
        X_in_stack = get_X_in_stack(self.N_A, self.sqrt_n_a, action_list_array, S_array_2d)
        Qsa = self.QSA_net[P_no-1](X_in_stack).numpy()[:,0]
        action_prob = list(Qsa)
        return action_prob

    def learning_cnndqn_variable_epsilon(self, N_episodes=2, ff=0.9, lr=0.01, 
            epsilon_d={'first value':0.4, 'epsilon change episode': 000, 'second value':0.1}, 
            print_cnt=10):
        cnt = [0, 0, 0, 0, 0, 0, 0, 0, 0] # tie, p1, p2
        cnt_trace = [cnt.copy()]        

        # Opponent player index
        P_no = 1 # player Q function, regardless of play order (first or next)
        play_order = 1
        ttt_env = Tictactoe_Env(self.N_A, play_order=play_order) #both X but start 1st and 2nd
        optimizer = tf.keras.optimizers.Adam()
        loss_f = tf.keras.losses.MeanSquaredError()   

        self.epsilon = epsilon_d['first value']
        for episode in range(N_episodes):
            if episode == epsilon_d['epsilon change episode']:                           
                self.epsilon = epsilon_d['second value']

            S, _ = ttt_env.reset(play_order=play_order)
            done = False            
            Replay_buff = []
            while not done:
                # self.epsilon = epsilon # epsilon is a hyperparamter for exploration
                action, _ = self.get_action(P_no, S)
                S_new, _, reward, done = ttt_env.step(action)
                Replay_buff.append([S.copy(), action, S_new.copy(), reward, done])
                # print(episode, [S, action, S_new, reward])
                S = S_new

            #######################################
            # DQN start, here for learning   
            #######################################
            # print('play_order, P_no = ', play_order, P_no)
            # No buffer shuffling
            sqrt_n_a = int(np.sqrt(self.N_A))
            X_N = np.zeros((len(Replay_buff), 2, sqrt_n_a, sqrt_n_a))
            y_N = np.zeros(len(Replay_buff))
            for i, buff in enumerate(Replay_buff):
                S, action, S_new, reward, done = buff
                if done:
                    y = reward
                else:
                    action_list_new, _ = self.find_action_list(S_new)
                    action_prob_array_new = self.get_q_net(S_new, action_list_new, P_no)
                    y = reward + ff*np.max(action_prob_array_new)
                y_N[i] = y
                action_buff = [0] * self.N_A
                action_buff[action] = 1
                X_N[i] = self.make_X_in(S, action_buff)

            with tf.GradientTape() as tape:
                Qsa_N = self.QSA_net[P_no-1](X_N)
                loss_value = loss_f(y_N, Qsa_N)
            gradients = tape.gradient(loss_value, self.QSA_net[P_no-1].trainable_weights)
            optimizer.apply_gradients(zip(gradients, self.QSA_net[P_no-1].trainable_weights))

            if Replay_buff[-1][3] == 1.0: 
                cnt[1] += 1               # P_no = 1 
                cnt[2 + play_order] += 1  # player_order = 1
                cnt[4 + play_order] += 1  # player_order | P_no = 1, so it occupied 2 lists
            elif Replay_buff[-1][3] == 0.5:
                cnt[0] += 1 
            else: # play_order = 2
                cnt[2] += 1                   # P_no = 2  
                cnt[2 + 3 - play_order] += 1  # player_order = 2
                cnt[6 + 3 - play_order] += 1  # player_order | P_no = 2, so it start from 6

            cnt_trace.append(cnt.copy())

            if episode % print_cnt == 0:
                print()
                print('Finished episode: #', episode)
                print('cnt: ', cnt)
                print(f'Play order:{play_order}, P_no:{P_no}')
                print('S = [0,0,0, 0,0,0, 0,0,0]')
                S = [0] * self.N_A
                Qsa_0_0, Qsa_1_0 = [], []
                for action in range(self.N_A):
                    action_buff = [0] * self.N_A
                    action_buff[action] = 1
                    X_in = np.zeros((1, 2, sqrt_n_a, sqrt_n_a))
                    X_in[0] = self.make_X_in(S, action_buff)
                    Qsa_0_0.append(self.QSA_net[0](X_in).numpy()[0,0])
                    Qsa_1_0.append(self.QSA_net[1](X_in).numpy()[0,0])
                print('Qsa[0][0,:]', [f'{Qsa_0_0[a]:.1e}' for a in range(9)])
                print('Qsa[1][0,:]', [f'{Qsa_1_0[a]:.1e}' for a in range(9)])
                print('Exproration: Epsilon=', self.epsilon)

            play_order = 3 - play_order # 1 --> 2, 2 --> 1

        return cnt_trace        
   
    def _learning_cnndqn_variable_epsilon_active(self, N_episodes=2, ff=0.9, lr=0.01, 
            epsilon_d={'first value':0.4, 'epsilon change episode': 000, 'second value':0.1}, 
            print_cnt=10, agent_type:int=0):
        """
        The computer player is an agent playing better than a random policy player and 
        reach to the human best player. 
        """
        cnt = [0, 0, 0, 0, 0, 0, 0, 0, 0] # tie, p1, p2
        cnt_trace = [cnt.copy()]        

        # Opponent player index
        P_no = 1 # player Q function, regardless of play order (first or next)
        play_order = 1
        ttt_env = Tictactoe_Env_Active(self.N_A, play_order=play_order, agent_type=agent_type) #both X but start 1st and 2nd
        optimizer = tf.keras.optimizers.Adam()
        loss_f = tf.keras.losses.MeanSquaredError()   

        self.epsilon = epsilon_d['first value']
        for episode in range(N_episodes):
            if episode == epsilon_d['epsilon change episode']:                           
                self.epsilon = epsilon_d['second value']

            S, _ = ttt_env.reset(play_order=play_order)
            done = False            
            Replay_buff = []
            while not done:
                # self.epsilon = epsilon # epsilon is a hyperparamter for exploration
                action, _ = self.get_action(P_no, S)
                S_new, _, reward, done = ttt_env.step(action)
                Replay_buff.append([S.copy(), action, S_new.copy(), reward, done])
                # print(episode, [S, action, S_new, reward])
                S = S_new

            #######################################
            # DQN start, here for learning   
            #######################################
            # print('play_order, P_no = ', play_order, P_no)
            # No buffer shuffling
            sqrt_n_a = int(np.sqrt(self.N_A))
            X_N = np.zeros((len(Replay_buff), 2, sqrt_n_a, sqrt_n_a))
            y_N = np.zeros(len(Replay_buff))
            for i, buff in enumerate(Replay_buff):
                S, action, S_new, reward, done = buff
                if done:
                    y = reward
                else:
                    action_list_new, _ = self.find_action_list(S_new)
                    action_prob_array_new = self.get_q_net(S_new, action_list_new, P_no)
                    y = reward + ff*np.max(action_prob_array_new)
                y_N[i] = y
                action_buff = [0] * self.N_A
                action_buff[action] = 1
                X_N[i] = self.make_X_in(S, action_buff)
            with tf.GradientTape() as tape:
                Qsa_N = self.QSA_net[P_no-1](X_N)
                loss_value = loss_f(y_N, Qsa_N)
            gradients = tape.gradient(loss_value, self.QSA_net[P_no-1].trainable_weights)
            optimizer.apply_gradients(zip(gradients, self.QSA_net[P_no-1].trainable_weights))

            if Replay_buff[-1][3] == 1.0: 
                cnt[1] += 1               # P_no = 1 
                cnt[2 + play_order] += 1  # player_order = 1
                cnt[4 + play_order] += 1  # player_order | P_no = 1, so it occupied 2 lists
            elif Replay_buff[-1][3] == 0.5:
                cnt[0] += 1 
            else: # play_order = 2
                cnt[2] += 1                   # P_no = 2  
                cnt[2 + 3 - play_order] += 1  # player_order = 2
                cnt[6 + 3 - play_order] += 1  # player_order | P_no = 2, so it start from 6

            cnt_trace.append(cnt.copy())

            if episode % print_cnt == 0:
                print()
                print('Finished episode: #', episode)
                print('cnt: ', cnt)
                print(f'Play order:{play_order}, P_no:{P_no}')
                print('S = [0,0,0, 0,0,0, 0,0,0]')
                S = [0] * self.N_A
                Qsa_0_0, Qsa_1_0 = [], []
                for action in range(self.N_A):
                    action_buff = [0] * self.N_A
                    action_buff[action] = 1
                    X_in = np.zeros((1, 2, sqrt_n_a, sqrt_n_a))
                    X_in[0] = self.make_X_in(S, action_buff)
                    Qsa_0_0.append(self.QSA_net[0](X_in).numpy()[0,0])
                    Qsa_1_0.append(self.QSA_net[1](X_in).numpy()[0,0])
                print('Qsa[0][0,:]', [f'{Qsa_0_0[a]:.1e}' for a in range(9)])
                print('Qsa[1][0,:]', [f'{Qsa_1_0[a]:.1e}' for a in range(9)])
                print('Exproration: Epsilon=', self.epsilon)

            play_order = 3 - play_order # 1 --> 2, 2 --> 1

        return cnt_trace  

    def learning_cnndqn_variable_epsilon_active(self, N_episodes=2, ff=0.9, lr=0.01, 
            epsilon_d={'first value':0.4, 'epsilon change episode': 000, 'second value':0.1}, 
            print_cnt=10, agent_type:int=0):
        """
        The computer player is an agent playing better than a random policy player and 
        reach to the human best player. 
        """
        cnt = [0, 0, 0, 0, 0, 0, 0, 0, 0] # tie, p1, p2
        cnt_trace = [cnt.copy()]        

        # Opponent player index
        P_no = 1 # player Q function, regardless of play order (first or next)
        play_order = 1
        ttt_env = Tictactoe_Env_Active(self.N_A, play_order=play_order, 
                agent_type=agent_type, Q_System_Class=Q_System_CNNDQN)
        optimizer = tf.keras.optimizers.Adam()
        loss_f = tf.keras.losses.MeanSquaredError()   

        self.epsilon = epsilon_d['first value']
        for episode in range(N_episodes):
            if episode == epsilon_d['epsilon change episode']:                           
                self.epsilon = epsilon_d['second value']

            S, _ = ttt_env.reset(play_order=play_order)
            done = False            
            Replay_buff = []
            while not done:
                # self.epsilon = epsilon # epsilon is a hyperparamter for exploration
                action, _ = self.get_action(P_no, S)
                S_new, _, reward, done = ttt_env.step(action)
                Replay_buff.append([S.copy(), action, S_new.copy(), reward, done])
                # print(episode, [S, action, S_new, reward])
                S = S_new

            #######################################
            # DQN start, here for learning   
            #######################################
            # print('play_order, P_no = ', play_order, P_no)
            # No buffer shuffling
            sqrt_n_a = int(np.sqrt(self.N_A))
            X_N = np.zeros((len(Replay_buff), 2, sqrt_n_a, sqrt_n_a), dtype='float32')
            y_N = np.zeros(len(Replay_buff), dtype='float32')
            for i, buff in enumerate(Replay_buff):
                S, action, S_new, reward, done = buff
                if done:
                    y = reward
                else:
                    action_list_new, _ = self.find_action_list(S_new)
                    action_prob_array_new = self.get_q_net(S_new, action_list_new, P_no)
                    y = reward + ff*np.max(action_prob_array_new)
                y_N[i] = y
                action_buff = [0] * self.N_A
                action_buff[action] = 1
                X_N[i] = self.make_X_in(S, action_buff)

            cnn_learning_inline(self.QSA_net[P_no-1], X_N, y_N, loss_f, optimizer)    

            if Replay_buff[-1][3] == 1.0: 
                cnt[1] += 1               # P_no = 1 
                cnt[2 + play_order] += 1  # player_order = 1
                cnt[4 + play_order] += 1  # player_order | P_no = 1, so it occupied 2 lists
            elif Replay_buff[-1][3] == 0.5:
                cnt[0] += 1 
            else: # play_order = 2
                cnt[2] += 1                   # P_no = 2  
                cnt[2 + 3 - play_order] += 1  # player_order = 2
                cnt[6 + 3 - play_order] += 1  # player_order | P_no = 2, so it start from 6

            cnt_trace.append(cnt.copy())

            if episode % print_cnt == 0:
                print()
                print('Finished episode: #', episode)
                print('cnt: ', cnt)
                print(f'Play order:{play_order}, P_no:{P_no}')
                print('S = [0,0,0, 0,0,0, 0,0,0]')
                S = [0] * self.N_A
                Qsa_0_0, Qsa_1_0 = [], []
                for action in range(self.N_A):
                    action_buff = [0] * self.N_A
                    action_buff[action] = 1
                    X_in = np.zeros((1, 2, sqrt_n_a, sqrt_n_a), dtype='float32')
                    X_in[0] = self.make_X_in(S, action_buff)
                    Qsa_0_0.append(self.QSA_net[0](X_in).numpy()[0,0])
                    Qsa_1_0.append(self.QSA_net[1](X_in).numpy()[0,0])
                print('Qsa[0][0,:]', [f'{Qsa_0_0[a]:.1e}' for a in range(9)])
                print('Qsa[1][0,:]', [f'{Qsa_1_0[a]:.1e}' for a in range(9)])
                print('Exproration: Epsilon=', self.epsilon)

            play_order = 3 - play_order # 1 --> 2, 2 --> 1

        return cnt_trace  

def learning_stage_cnndqn_variable_epsilon(N_episodes=100, 
        epsilon_d={'first value':0.4, 'epsilon change episode': 000, 'second value':0.1}, 
        agent_type:int=0, save_flag=True, fig_flag=False):
    ff = 0.9
    lr = 0.01
    N_Symbols = 3 # 0=empty, 1=plyaer1, 2=player2
    N_A = 9 # (0,0), (0,1), ..., (2,2)
    print_cnt = N_episodes / 10

    my_Q_System = Q_System_CNNDQN(N_A, N_Symbols)
    cnt_trace = my_Q_System.learning_cnndqn_variable_epsilon_active(N_episodes=N_episodes, 
                    ff=ff, lr=lr, epsilon_d=epsilon_d, print_cnt=print_cnt,
                    agent_type=agent_type)

    print('-------------------')
    cnt_last = cnt_trace[-1]
    cnt_last_normal = np.array(cnt_last) / np.sum(cnt_last[0:3])
    # Showing normalized counts as well so as to make feel the progress.
    # In random agent playing here, no progress should be displayed.
    print(N_episodes, f"Last cnt:{cnt_last}, Normalized last cnt:{cnt_last_normal}")

    if save_flag:        
        my_Q_System.save()

    if fig_flag:
        plot_cnt_trace_normal_order_detail(cnt_trace, title='CNN-DQN')

    return my_Q_System


def q1_playing():
    print()
    print('Loading the trained agent...')
    Q2 = input_default('Do you want to start first?(0=yes,1=no,default=0) ', 0, int)
    player_human = Q2 + 1
    if player_human == 1:
        print('You=1(X), Agent=2(O)') 
    else:
        print('Agent=1(X), You=2(O)') 
    
    class_name = input_default('Which agent do you want to play with?(0=Q-learn,1=DQN,2=CNN-DQN,default=2)', 2, int)

    N_A = 9
    N_Symbols = 3
    if class_name == 0:
        trained_Q_System = Q_System(None)
    elif class_name == 1:
        trained_Q_System = Q_System_DQN(N_A=N_A, N_Symbols=N_Symbols)
    else: # class_name == 2 (default)
        trained_Q_System = Q_System_CNNDQN(N_A=N_A, N_Symbols=N_Symbols)

    trained_Q_System.load()
    print(f'You are going to playing with our AI agent - {trained_Q_System.class_name}.')
    trained_Q_System.play_with_human_inference(player_human)


def q1_learning():
    """
    1. After learning testing playing with a random playing agent and a best playing agent. 
       In the test, our agent should not include rule playing while the best player will has it.
       The best player is equal to out agent with the guide rule function.
    2. INcluding the learning method to learn with expert not random player. 
    """
    print()
    print('------------------------------')
    print('Start to learn a new agent...')
    print()
    N_episodes = input_default('How many episode do you want to learn?(default=10000) ', 10000, int)

    print()
    epsilon = input_default('What is initial Epsilon for exploration?(default=0.4) ', 0.4, float)
    second_epsilon = input_default('What is second Epsilon for exploration?(default=0.1) ', 0.1, float)
    epsilon_change_episode = input_default('What episode do you want to change Epsilon?(default=2000) ', 2000, int)
    agent_type = input_default('Which type of a computer agent do you want to play with?(default=0:random, 1:advanced, 2:premium) ', 0, int)
    
    print()
    print('0) MC Backup with e-Greedy')
    print('1) Q-learning')
    print('2) Q-learning with variable esplison')
    print('3) DQN with variable esplison')
    print('4) CNN-DQN with variable esplison')
    Method = input_default('What learning method do you want to use?(0=default) ', 0 , int)
    if Method == 0:
        _ = learning_stage_mc(N_episodes=N_episodes, fig_flag=True)
    elif Method == 1:
        _ = learning_stage_qlearn(N_episodes=N_episodes, epsilon=epsilon, fig_flag=True)
    elif Method == 2:
        epsilon_d = {'first value':epsilon, 'second value':second_epsilon, 'epsilon change episode':epsilon_change_episode}
        _ = learning_stage_qlearn_variable_epsilon(N_episodes=N_episodes, epsilon_d=epsilon_d, fig_flag=True)
    elif Method == 3:
        epsilon_d = {'first value':epsilon, 'second value':second_epsilon, 'epsilon change episode':epsilon_change_episode}
        _ = learning_stage_dqn_variable_epsilon(N_episodes=N_episodes, epsilon_d=epsilon_d, 
                agent_type=agent_type, fig_flag=True)
    elif Method == 4:
        epsilon_d = {'first value':epsilon, 'second value':second_epsilon, 'epsilon change episode':epsilon_change_episode}
        _ = learning_stage_cnndqn_variable_epsilon(N_episodes=N_episodes, epsilon_d=epsilon_d, 
                agent_type=agent_type, fig_flag=True)
    else:
        print('No such method is supported.')


def modeling_stage_qlearn(N_episodes=100, fig_flag=False):
    N_Symbols = 3 # 0=empty, 1=plyaer1, 2=player2
    N_A = 9 # (0,0), (0,1), ..., (2,2)

    my_Q_System = Q_System_QL(N_A, N_Symbols)
    my_Q_System.modeling_qlearn(N_episodes=N_episodes)

    return my_Q_System


def _q1_modeling():
    """
    1. After learning testing playing with a random playing agent and a best playing agent. 
       In the test, our agent should not include rule playing while the best player will has it.
       The best player is equal to out agent with the guide rule function.
    2. INcluding the learning method to learn with expert not random player. 
    """
    print()
    print('------------------------------')
    print('Start to model the playing system')
    print()
    N_episodes = input_default('How many episode do you want for modeling?(default=10000) ', 10000, int)

    my_Q_System = modeling_stage_qlearn(N_episodes=N_episodes, fig_flag=True)

    print()
    print('Checking modeling results')    
    Done = False
    while not Done:
        S_str = input_default('Enter State vector (S)?(defulat=[0,0,0,0,0,0,0,0,0]) ', '[0,0,0,0,0,0,0,0,0]', str)
        S = eval(S_str)
        action = input_default('Enter action?(defulat=0) ', 0, int)
        S_idx_base = np.power(3, range(9))
        S_idx = np.sum(S_idx_base * S)
        print(f'N_Model[{S_idx},:] = ', my_Q_System.N_Model[S_idx,:])
        print(f'Model_R[{S_idx},:] = ', [my_Q_System.get_Model_R(S_idx,i) for i in range(9)])
        print(f'Model_P[{S_idx},{action},:] = ', [my_Q_System.get_Model_P(S_idx,action,i) for i in range(9)])
        yn = input_default('Quit=0 or test again=1?(defalut=0) ', 0, int)
        Done = not yn


def q1_modeling():
    """
    1. After learning testing playing with a random playing agent and a best playing agent. 
       In the test, our agent should not include rule playing while the best player will has it.
       The best player is equal to out agent with the guide rule function.
    2. INcluding the learning method to learn with expert not random player. 
    """
    print()
    print('------------------------------')
    print('Start to model the playing system')
    print()
    N_episodes = input_default('How many episode do you want for modeling?(default=10000) ', 10000, int)

    my_Q_System = modeling_stage_qlearn(N_episodes=N_episodes, fig_flag=True)

    print()
    print('Checking modeling results')    
    Done = False
    while not Done:
        S_str = input_default('Enter State vector (S)?(defulat=[0,0,0,0,0,0,0,0,0]) ', '[0,0,0,0,0,0,0,0,0]', str)
        S = eval(S_str)
        action = input_default('Enter action?(defulat=0) ', 0, int)
        S_idx_base = np.power(3, range(9))
        S_idx = np.sum(S_idx_base * S)
        print(f'N_Model[{S_idx},:] = ', my_Q_System.N_Model[S_idx,:])
        print(f'Model_R[{S_idx},:] = ', [my_Q_System.get_Model_R(S_idx,i) for i in range(9)])
        print(f'Model_P[{S_idx},{action},:] = ', [my_Q_System.get_Model_P(S_idx,action,i) for i in range(9)])
        print(f'Model_Done1[{S_idx},:] = ', [my_Q_System.get_Model_Done1(S_idx,i) for i in range(9)])
        print(f'Model_Done2[{S_idx},{action},:] = ', [my_Q_System.get_Model_Done2(S_idx,action,i) for i in range(9)])
        yn = input_default('Quit=0 or test again=1?(defalut=0) ', 0, int)
        Done = not yn

def q1_testing():
    print('Start to test code...')
    print()
    Q1 = input_default('How many episode do you want to play?(default=10000) ', 10000, int)
    
    print()
    print('0) Play by two random players')
    print('1) Play for dqn player and random player')
    Q2 = input_default('Which playing do you want?(default=0) ', 0, int)

    if Q2 == 0:
        print('Playing between two random players')
        _ = playing_stage_random(N_episodes=Q1, fig_flag=True)
    else:
        print('Playing between our dqn player and a random player')
        playing_stage_dqn(N_episodes=Q1, fig_flag=True)



def q1_dyna(): 
    """
    Learning and Planning
    """
    print()
    print('------------------------------')
    print('Start to learn and plain a new agent...')
    print()
    N_episodes = input_default_with('How many episode do you want to learn?', 10000, int) #10000
    N_plan = input_default_with('How many plan times do you plan?', 10, int)

    print()
    epsilon = input_default('What is initial Epsilon for exploration?(default=0.4) ', 0.4, float)
    second_epsilon = input_default('What is second Epsilon for exploration?(default=0.1) ', 0.1, float)
    epsilon_change_episode = input_default('What episode do you want to change Epsilon?(default=2000) ', 2000, int)
    # agent_type = input_default('Which type of a computer agent do you want to play with?(default=0:random, 1:advanced, 2:premium) ', 0, int)
    
    print()
    print('0) Q-learning with variable esplison')
    Method = input_default('What learning method do you want to use?(0=default) ', 0 , int)
    if Method == 0:
        epsilon_d = {'first value':epsilon, 'second value':second_epsilon, 'epsilon change episode':epsilon_change_episode}
        _ = learning_planning_stage_qlearn_variable_epsilon(N_episodes=N_episodes, epsilon_d=epsilon_d, N_plan=N_plan, fig_flag=True)
    else:
        print('No such method is supported.')

def tictactoe_main():
    """
    Three types of starting questions will be given to a user.
    """
    Done = False
    while not Done:
        print()
        print('== Start TicTacToe Aget Framework ==')
        print('- Developed by Sungjin Kim, 2020')
        print()
        print('0) Playing a game')
        print('1) Learning a new agent')
        print('2) Testing code')
        print('3) Modeling')
        print('4) Dyna - Learning & Plaining')
        print('999) Quit')
        Q1 = input_default_with('What do you want?', 999, int)
        if Q1 == 0:
            q1_playing()
        elif Q1 == 1:
            q1_learning()
        elif Q1 == 2:
            q1_testing()
        elif Q1 == 3:
            q1_modeling()
        elif Q1 == 4:
            q1_dyna()
        elif Q1 == 999:
            Done = True
        else:
            print('Type a different option listed in the above table.')


def main():
    """
    Futher than Tictactoe applications are considered, including random walk, maze, etc. 
    """
    Done = False
    while not Done:
        print()
        print('==RL Aget Testing Framework ==')
        print('- Developed by Sungjin Kim, 2020')
        print()
        print('0) Tictactoe')
        print('1) Random walk')
        print('999) Quit')
        Q1 = input_default_with('Which application do you want to perform?', 999, int)
        if Q1 == 0:
            tictactoe_main()
        elif Q1 == 1:
            randomwalk_main()
        elif Q1 == 999:
            Done = True
        else:
            print('Type a different option listed in the above table.')            


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


def randomwalk_main():
    print('Testing random walk')
    N_episodes = input_default_with('How many episodes do you want to run?', 10)
    rl_mode_index = input_default_with('Which mode do you want to use?(0=td,1=mc,2=pg_learning)', 0)
    rl_mode = ['td', 'mc', 'pg'][rl_mode_index]
    if rl_mode == 'pg':
        randomwalk_run_pg_learning(N_episodes=N_episodes)
    else:
        randomwalk_run(N_episodes=N_episodes, rl_mode=rl_mode)


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
        performance = np.log(prob) * discounted_return

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
        
        # Once tf is chagned to numpy, no gradient is able to be calculated.
        performance_vec = tf.math.log(prob) * discounted_return.reshape(-1,1)
        #print(prob)
        #print(discounted_return.reshape(-1,))
        #print(performance_vec)

        performance = tf.reduce_sum(performance_vec)
        gradients = tape.gradient(-performance, self.function_approx.trainable_weights)
        optimizer.apply_gradients(zip(gradients, self.function_approx.trainable_weights))

    def update_mc(self, replay_buff_d):
        self.update_Qsa_mc(replay_buff_d)
        self.update_Vs_mc(replay_buff_d)


def randomwalk_run_pg_learning(N_episodes=1, learning_mode=True):
    print('Plocy gradient mode (Under development)')
    N_S = 5
    N_A = 2
    rl_system = RL_System_PG_TF(N_S, N_A)
    random_walk_env = RandomWalkEnv()
    S = random_walk_env.reset()

    optimizer = tf.keras.optimizers.Adam()
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

if __name__ == "__main__":
    # This is the main function.
    main()


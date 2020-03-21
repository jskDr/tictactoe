import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pickle
import numba
from numba import jit

# TicTacToe game has nine stateus with nine actions. An user can put his ston on any postion in the borad except 

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


def buff_depart(Buff, disp_flag=False):
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
        obj = [self.N_A, self.N_Symbols, self.epsilon, self.Qsa]
        pickle.dump(obj, f)
        f.close()

    def load(self):
        f = open('tictactoe_data.pckl', 'rb')
        obj = pickle.load(f)
        [self.N_A, self.N_Symbols, self.epsilon, self.Qsa] = obj
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
            reward = calc_reward_tf(Sa)
            # print('Checking available win:P_no, S, Sa, action', P_no, S, Sa, action)
            if reward:
                print('Founded available win: Sa, action-->', Sa, action)
                return True, action
        return False, None

    def check_lose_protection_action(self, P_no, S, action_list):
        """
        yes, action = check_lose_protection_action(P_no, S, action_list)
        - check whether there is need for the imidiate action to protect lose
        by the proponent's next move.
        """ 
        for action in action_list:
            Sa = S.copy()
            player_op = 1 if P_no==2 else 2
            Sa[action] = player_op
            reward = calc_reward_tf(Sa)
            if reward:
                print('Founded lose protection: Sa, action -->', Sa, action)
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
        if action_idx == len(action_prob): # if all zeros in actoin_prob
            action = action_list[tf.squeeze(np.random.randint(0, len(action_list), 1))]
        else:
            action = action_list[action_idx]
        if self.disp_flag:
            print('S_idx', S_idx, 'action', action, 'action_list', action_list, 'action_prob', action_prob)
        return action
    
    def policy(self, P_no, S, action_list):
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

    def find_action_list(self, S):
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
        """Return action, done
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
        S_idx = self.calc_S_idx(S)                    
        print('[Agent-Qsa]', [f'{self.Qsa[P_no-1][S_idx,a]:.1e}' for a in action_list])
        print('Agent action:', action)
        done = no_occupied == (self.N_A - 1)
        return action, done

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
        S_idx = self.calc_S_idx(S)
        print('[Qsa]', [f'{a}:{self.Qsa[P_no-1][S_idx,a]:.1e}' for a in action_list])

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
            
            win_player = calc_reward_tf(S)
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
            
            win_player = calc_reward_tf(S)
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

    def play_by_scenario(self, P_no=1, action_list=[4, 1, 3, 5, 6, 2, 0]):
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
            
            win_player = calc_reward_tf(S)
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
            
            win_player = calc_reward_tf(S)
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
        return cnt_trace


def plot_cnt_trace(cnt_trace):
    N_cnt = len(cnt_trace)
    cnt_d = {'Equal':np.zeros(N_cnt,dtype=int), 'P1':np.zeros(N_cnt,dtype=int), 'P2':np.zeros(N_cnt,dtype=int)}
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
    plt.title('Learned (P#1) vs. Random (P#2) policies during learning')
    plt.show(True)


def learning_stage(N_episodes=100, save_flag=True, fig_flag=False):
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
        plot_cnt_trace(cnt_trace)

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
        win_player = calc_reward_tf(S)
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

def get_Vs(Qsa, Saa, action_list, N_A=9, N_Symbols=3, disp_flag=False):
    Vs = 0.
    for action_pro in range(N_A):
        if action_pro not in action_list:
            Saa_Idx = calc_S_idx_numba(Saa, N_Symbols=N_Symbols)
            Vs += Qsa[Saa_Idx, action_pro]
            if disp_flag:
                print(f'Qsa[{Saa}, {action_pro}] = ', Qsa[Saa_Idx, action_pro])
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
    
    def update(self, action_list=[], max_actions=1, disp_flag=False): 
        """This will update
        """
        # Bellman_exp_inplace(self.Qsa, S, action, action_list, ff=0.9, N_A=9, N_Symbols=3):
        
        # This is testing code
        # print('Top: action_list:', action_list)
        if len(action_list) > max_actions:
            # print('Max action stop: action_list:', action_list)
            return 
        
        # P_no = 1
        if len(action_list) % 2 == 1: # a stage for player 2 to play
            for action in range(self.N_A):
                if action not in action_list:
                    action_list.append(action)
                    Sa = action_list_2_state(action_list)
                    done = False if calc_reward_numba(MASK_L, Sa) == 0 else True
                    if not done:
                        # print('Enter: action_list:', action_list)
                        self.update(action_list, max_actions=max_actions)
                    else:
                        print('Player 2 wins with Sa, action_list and done', Sa, action_list, done)
                    action_list.pop()  
        else: # player 1
            S = action_list_2_state(action_list)
            for action in range(self.N_A):
                if action not in action_list:
                    done = Bellman_exp_inplace(self.Qsa, S, action, action_list, ff=self.ff, N_A=self.N_A, N_Symbols=self.N_Symbols, disp_flag=disp_flag)    
                    if not done:
                        action_list.append(action)
                        # print('Enter: action_list:', action_list)
                        self.update(action_list, max_actions=max_actions)
                        action_list.pop()  


def Bellman_exp_inplace(Qsa, S, action, action_list, ff=0.9, N_A=9, N_Symbols=3, disp_flag=False):
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
                    D_E_V_Saa = get_Vs(Qsa, Saa, action_list, N_A=N_A, N_Symbols=N_Symbols, disp_flag=disp_flag)
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
        test = [calc_reward_tf(S) for S in S_examples] 
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


def main():
    Q1 = input_default('1. Loading a trained agent (0) or Learning a new agent (1)? (default=0) ', 0, int)
    if Q1 == 0:
        print('Loading the trained agent...')
        Q2 = input_default('2. Do you want to start first?(0=yes,1=no,default=0) ', 0, int)
        player_human = Q2 + 1
        if player_human == 1:
            print('You=1, Agent=2') 
        else:
            print('Agent=1, You=2') 
        trained_Q_System = Q_System(None)
        trained_Q_System.load()
        trained_Q_System.play_with_human(player_human)
        # print(len(trained_Q_System.Qsa))
    else:
        print('Start to learn a new agent...')
        Q2 = input_default('2. How many episode do you want to learn?(default=10000) ', 10000, int)
        # my_Q_System = learning_stage(N_episodes=Q2, fig_flag=True)
        _ = learning_stage(N_episodes=Q2, fig_flag=True)
        # print(len(my_Q_System.Qsa))


if __name__ == "__main__":
    main()
    # Testing('all')
    pass

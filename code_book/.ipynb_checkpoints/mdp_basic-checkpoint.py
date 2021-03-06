import numpy as np

class MDP:
    def __init__(self):
        self.init_policy_Rsa()
        
    def init_policy_Rsa(self):
        # policy[state][action] = 0.9 (<-- prob)
        # None state is the last state        
        self.policy_actions_table = [['Facebook', 'Quit'], ['Facebook', 'Study'], ['Sleep', 'Study'], ['Pub', 'Study'], None]
        self.Rsa = [[-1,0], [-1,-2], [0, -2], [1,10]]
        self.N_states = len(self.policy_actions_table)
        self.policy = []
        for actions in self.policy_actions_table:
            if actions:
                N_actions = len(actions)
                self.policy.append(np.ones(N_actions)/N_actions)
                
        # policy가 고려되지 않은 관계임. Policy에 따른 가중에 별도로 고려되어야 함.
        self.Psas = np.zeros([self.N_states, 2, self.N_states]) # Probability
        self.Psas[0,0,0], self.Psas[0,1,1] = 1.0, 1.0
        self.Psas[1,0,0], self.Psas[1,1,2] = 1.0, 1.0
        self.Psas[2,0,4], self.Psas[2,1,3] = 1.0, 1.0
        self.Psas[3,0,1], self.Psas[3,0,2], self.Psas[3,0,3], self.Psas[3,1,4] = 0.2, 0.4, 0.4, 1.0
        
        """
        Policy를 
        self.Psas = []
        # state = 0
        self.Psas.append([{0:1.0}, {1:1.0}]) # action=[0,1]
        # state = 1
        self.Psas.append([{0:1.0}, {2:1.0}]) # action=[0,1]
        # state = 2
        self.Psas.append([{4:1.0}, {3:1.0}]) # action=[0,1]
        # state = 3
        self.Psas.append([{1:0.2, 2:0.4, 3:0.4}, {4:1.0}]) # action=[0,1]        
        """        
        
    def init_v(self):
        self.v = np.zeros(self.N_states)
        
    def init_q(self):
        self.q = []
        for s in range(self.N_states - 1):
            self.q.append(np.zeros(len(self.policy[s])))
        
    def calc_bellman_v(self, s):
        return bellman_v(self.v, self.policy, self.Rsa, self.Psas, s=s)    
    
    def calc_bellman_q(self, s, a):
        #return 0
        return bellman_q(self.q, self.policy, self.Rsa, self.Psas, s=s, a=a)    
    
    def get_v(self, N_iter=10):
        self.init_v()
        for n in range(N_iter):
            for s in range(self.N_states-1):
                self.v[s] = (self.v[s] * n + self.calc_bellman_v(s))/(n+1)        
        
        for s in range(self.N_states):
            print(f'v[{s}]={self.v[s]}')
        return self.v
    
    def get_q(self, N_iter=10):
        self.init_q()
        
        for n in range(N_iter):
            for s in range(self.N_states-1):
                for a in range(len(self.policy[s])):
                    #print(f'[?]s,a={s,a} --> {self.q[s][a]}')
                    self.q[s][a] = (self.q[s][a] * n + self.calc_bellman_q(s,a))/(n+1)  
                    #self.q[s][a] = (self.q[s][a] * n)/(n+1) 
        
        for s in range(self.N_states-1):
            for a in range(len(self.policy[s])):
                print(f'q[{s},{a}]={self.q[s][a]}')
        return self.q        
    
    def test(self):
        print(f'Policy: {self.policy}')
        self.get_v()
        self.get_q()
        
        
def bellman_v(v, policy, Rsa, Psas, s:int=0, forgetting_factor:float=1.0):
    Gs = 0
    for a in range(len(policy[s])):
        Gs += policy[s][a] * bellman_q_by_v(v, policy, Rsa, Psas, s=s, a=a, forgetting_factor=forgetting_factor)
    return Gs

def bellman_q_by_v(v, policy, Rsa, Psas, s:int=0, a:int=0, forgetting_factor:float=1.0):
    reward = Rsa[s][a]
    v_next = 0
    for next_s in range(len(Psas[s,a])):
        if Psas[s,a,next_s] and next_s < len(v) - 1:
            v_next += Psas[s,a,next_s] * v[next_s]     
    Gs = reward + forgetting_factor * v_next
    return Gs

def bellman_q(q, policy, Rsa, Psas, s:int=0, a:int=0, forgetting_factor:float=1.0):
    reward = Rsa[s][a]
    v_next = 0
    for next_s in range(len(Psas[s,a])):
        if Psas[s,a,next_s] and next_s < len(q) - 1:
            v = 0
            for next_a in range(len(policy[next_s])):
                # print(f'next_s,next_a={next_s,next_a}')
                v += policy[next_s][next_a] * q[next_s][next_a]
            v_next += Psas[s,a,next_s] * v     
    Gs = reward + forgetting_factor * v_next
                
if __name__ == '__main__':
    MDP().test()

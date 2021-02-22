import numpy as np

class MDP:
    def __init__(self):
        self.init_policy()
        self.init_Rsa()
        self.init_v()
        
    def init_policy(self):
        # policy[state][action] = 0.9 (<-- prob)
        # None state is the last state
        self.policy_actions_table = [['Facebook', 'Quit'], ['Facebook', 'Study'], ['Sleep', 'Study'], ['Pub', 'Study'], None]
        self.N_states = len(self.policy_actions_table)
        self.policy = []
        for actions in self.policy_actions_table:
            if actions:
                self.policy.append(np.ones(len(actions))/len(actions))
            
    def init_v(self):
        self.v = np.zeros(self.N_states)        
        
    def calc_v(self):
        s0 = 0
        self.trace_mc(s0)
        
    def init_Rsa(self):
        pass
    
    def do_action(self, state, action):
        pass
        
    def trace_mc(self, s0):
        """
        Monte Carlo
        """
        T = {'state':[], 'reward':[]}
        T['state'].append(s0)
        T['reward'].append(0)
        
        p_array = policy[s0]
        # numpy.random.choice(a, size=None, replace=True, p=None)
        action = np.random.choise(range(len(p_array)), p=p_array)
        new_state, reward = self.do_acion(s0, action)
                
    def test(self):
        print(f'Policy: {self.policy}')
        print(f'v: {self.v}')        
        
MDP().test()

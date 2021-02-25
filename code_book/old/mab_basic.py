import numpy as np

class MDP:
    def __init__(self):
        self.init_policy_Rsa()
        self.init_v()
        
    def init_policy_Rsa(self):
        # policy[state][action] = 0.9 (<-- prob)
        # None state is the last state        
        self.policy_actions_table = [['Facebook', 'Quit'], ['Facebook', 'Study'], ['Sleep', 'Study'], ['Pub', 'Study'], None]
        self.Rsa = [[-1,0], [-1,-2], [0, -2], [1,10]]
        self.N_states = len(self.policy_actions_table)
        self.policy = []
        for actions in self.policy_actions_table:
            if actions:
                self.policy.append(np.ones(len(actions))/len(actions))
                
        # policy가 고려되지 않은 관계임. Policy에 따른 가중에 별도로 고려되어야 함.
        self.Psas = np.zeros([self.N_states, 2, self.N_states]) # Probability
        self.Psas[0,0,0], self.Psas[0,1,1] = 1.0, 1.0
        self.Psas[1,0,0], self.Psas[1,1,2] = 1.0, 1.0
        self.Psas[2,0,4], self.Psas[2,1,3] = 1.0, 1.0
        self.Psas[3,0,1], self.Psas[3,0,2], self.Psas[3,0,3], self.Psas[3,1,4] = 0.2, 0.4, 0.4, 1.0
            
    def init_v(self):
        self.v = np.zeros(self.N_states)        
        
    def calc_v(self):
        s0 = 0
        self.trace_mc(s0)        
    
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
       
    def get_v(self):
        calc_v(self.v, self.policy, self.Rsa, self.Psas, s=0)
        
    def test(self):
        print(f'Policy: {self.policy}')
        print(f'v: {self.v}')        
        #self.get_v()        
        
def calc_v(v, policy, Rsa, Psas, s:int=0, forgetting_factor=1.0):
    """
    이런 방식의 계산은 불가능하다. 왜냐하면 무한루프를 돌수 있기 때문이다. 
    확률이 낮은 곳이라도 전체의 가능성을 모두 계산하는 아래 방식은 반드시 그 부분에 들어가야 하기 때문에 계속해서 반복이 된다.
    확률을 Monte Carlo로 방식으로 넣도록 하면 돌아갈 수 있지 않을까?
    """
    Gs = 0
    for a in range(len(policy[s])): # for문이 아닌 확률에 의해 선택되어서 돌아가게 함.
        reward = Rsa[s][a]
        v_next = 0
        for next_s in range(len(Psas[s,a])): # for문이 아닌 확률에 의해 선택되어서 돌아가게 함.
            print(f'Psas[{s,a,next_s}]={Psas[s,a,next_s]}')
            if next_s == len(v):
                break
            elif Psas[s, a, next_s]:
                print(f'Psas[{s}, {a}, {next_s}] = {Psas[s, a, next_s]}')
                # v_next += Psas[s, a, next_s] * calc_v(v, policy, Rsa, Psas, next_s)
        v[s] += policy[s][a] * (reward + forgetting_factor * v_next)
        print(f'reward={reward}')
        print(f'policy[{s}][{a}]={policy[s][a]}')
        print(f'v[{s}] = {v[s]}')
        return v[s]
        
MDP().test()
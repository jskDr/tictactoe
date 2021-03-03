import numpy as np

# from scipy.stats import norm
class MAB_Env:
    def __init__(self, mean_array, std_array):
        self.mean_array = mean_array
        self.std_array = std_array
        assert len(mean_array) == len(std_array), 'The lengths of mean and std arries must be the same.'
        self.N = len(self.std_array)

    def get_reward(self, action):
        assert action >= 0 and action < self.N, 'action should be smaller than len(mean_array)'
        reward = np.random.randn() * self.std_array[action] + self.mean_array[action]
        return reward


def run_episodes(MeanStd, N_episodes=10):
    N_A = MeanStd.shape[0]
    Qa = np.zeros(N_A, dtype=np.float)
    Na = np.zeros(N_A, dtype=np.float)
    Ra = np.zeros(N_A, dtype=np.float)
    for _ in range(N_episodes):
        action = np.argmax(Qa)
        reward = np.random.randn() * MeanStd[action,1] + MeanStd[action,0]        
        Na[action] += 1
        Ra[action] += reward 
        Qa[action] = Ra[action] / Na[action]
    return Qa      


def main():
    mean_array = np.array([1.,2.,3], dtype=np.float)
    std_array = np.array([1.,1.,1.], dtype=np.float)
    mab_env = MAB_Env(mean_array, std_array)
    
    # Testing for reward for every actions
    #for action in range(mab_env.N):
    #    print(action, '-->', mab_env.get_reward(action))    

    MeanStd = np.array([[1,1],[2,1],[3,1]], dtype=np.float) # 0=mean, 1=std
    Qa = run_episodes(MeanStd, N_episodes=1000)
    print('Qa:', Qa)

if __name__ == "__main__":
    # This is the main function.
    main()
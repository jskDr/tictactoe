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


def input_default_with(str, defalut_value, dtype=int):
    answer = input(str+f"[default={defalut_value}] ")
    if answer == '':
        return defalut_value
    else:
        return dtype(answer)
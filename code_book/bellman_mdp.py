"""
두 클라스를 공통인 MDP에서 상속하게 만듬. 
L은 list를 이용하는 방식이고 PD는 pandas를 이용하는 방식임.
두 방식에 충돌이 나는 경우, 기반이 되는 MDP에는 공통 요소만 남기고 나머지는 MDP_L에 옮겨둘 예정임.
"""
import numpy as np
from typing import List
import pandas as pd


class MDP:
    def __init__(self):
        self.init_policy_rsa_psas()

    def init_policy_rsa_psas(self):
        # policy[state][action] = 0.9 (<-- prob)
        # None state is the last state        
        self.policy_actions_table = [['Facebook', 'Quit'], ['Facebook', 'Study'],
                                     ['Sleep', 'Study'], ['Pub', 'Study'], None]
        self.Rsa = [[-1, 0], [-1, -2], [0, -2], [1, 10]]
        self.N_states = len(self.policy_actions_table)
        self.N_actions_in_s = []
        self.policy = []
        for actions in self.policy_actions_table:
            if actions:
                N_actions = len(actions)
                self.N_actions_in_s.append(N_actions)
                self.policy.append(np.ones(N_actions) / N_actions)
            else:
                self.N_actions_in_s.append(0)

        # policy가 고려되지 않은 관계임. Policy에 따른 가중에 별도로 고려되어야 함.
        self.Psas = np.zeros([self.N_states, 2, self.N_states])  # Probability
        self.Psas[0, 0, 0], self.Psas[0, 1, 1] = 1.0, 1.0
        self.Psas[1, 0, 0], self.Psas[1, 1, 2] = 1.0, 1.0
        self.Psas[2, 0, 4], self.Psas[2, 1, 3] = 1.0, 1.0
        self.Psas[3, 0, 1], self.Psas[3, 0, 2], \
        self.Psas[3, 0, 3], self.Psas[3, 1, 4] = 0.2, 0.4, 0.4, 1.0

    def init_v(self):
        self.v = np.zeros(self.N_states)

    def init_q(self):
        self.q = []
        for s in range(self.N_states - 1):
            self.q.append(np.zeros(self.N_actions_in_s[s]))
        self.q.append(0)
        # print('q=', self.q)

    def get_v(self, N_iter: int = 10) -> np.ndarray:
        self.init_v()
        for n in range(N_iter):
            for s in range(self.N_states - 1):
                self.v[s] = (self.v[s] * n + self.bellman_v(s)) / (n + 1)

        for s in range(self.N_states):
            print(f'v[{s}]={self.v[s]}')
        return self.v

    def get_q(self, N_iter: int = 10) -> List:
        self.init_q()

        for n in range(N_iter):
            for s in range(self.N_states - 1):
                for a in range(self.N_actions_in_s[s]):
                    # print(f'[?]s,a={s,a} --> {self.q[s][a]}')
                    self.q[s][a] = (self.q[s][a] * n +
                                    self.bellman_q(s, a)) / (n + 1)
                    # self.q[s][a] = (self.q[s][a] * n)/(n+1)

        for s in range(self.N_states - 1):
            for a in range(self.N_actions_in_s[s]):
                print(f'q[{s}][{a}]={self.q[s][a]}')
        return self.q

    def bellman_v(self, s: int = 0, forgetting_factor: float = 1.0) -> float:
        Gs = 0
        for a in range(len(self.policy[s])):
            Gs += self.policy[s][a] * self.bellman_q_by_v(
                s=s, a=a, forgetting_factor=forgetting_factor)
        return Gs

    def bellman_q_by_v(self, s: int = 0, a: int = 0, forgetting_factor: float = 1.0) -> float:
        reward = self.Rsa[s][a]
        v_next = 0
        for next_s in range(len(self.Psas[s, a])):
            if self.Psas[s, a, next_s] and next_s < len(self.v) - 1:
                v_next += self.Psas[s, a, next_s] * self.v[next_s]
        Gs = reward + forgetting_factor * v_next
        return Gs

    def bellman_q(self, s: int = 0, a: int = 0, forgetting_factor: float = 1.0) -> float:
        reward = self.Rsa[s][a]
        v_next = 0
        for next_s in range(len(self.Psas[s, a])):
            if self.Psas[s, a, next_s] and next_s < len(self.q) - 1:
                v = 0
                for next_a in range(len(self.policy[next_s])):
                    v += self.policy[next_s][next_a] * self.q[next_s][next_a]
                v_next += self.Psas[s, a, next_s] * v
        Gs = reward + forgetting_factor * v_next
        return Gs

    def test(self, N_Iter: int = 100):
        print('### v(s)')
        self.get_v(N_Iter)
        print('### q(s,a)')
        self.get_q(N_Iter)


class MDP_PD(MDP):
    # def __init__(self):
    #    super().__init__()

    def set_policy_rsa_psas(self):
        """
        PD Based Bellman Eq. with the above MDP
        """
        S_df = pd.DataFrame({'S': [0, 1, 2, 3, 4]})
        self.policy = pd.DataFrame({'s': [0, 0, 1, 1, 2, 2, 3, 3], 'a': [0, 1, 0, 1, 0, 1, 0, 1],
                                    'pi': [1 / 2, 1 / 2, 1 / 2, 1 / 2, 1 / 2, 1 / 2, 1 / 2, 1 / 2]})
        self.Rsa = pd.DataFrame(
            {'s': [0, 0, 1, 1, 2, 2, 3, 3], 'a': [0, 1, 0, 1, 0, 1, 0, 1], 'R': [-1, 0, -1, -2, 0, -2, 1, 10]})
        self.Psas = pd.DataFrame({'s': [0, 0, 1, 1, 2, 2, 3, 3, 3, 3], 'a': [0, 1, 0, 1, 0, 1, 0, 0, 0, 1],
                                  'next_s': [0, 1, 0, 2, 4, 3, 1, 2, 3, 4],
                                  'P': [1, 1, 1, 1, 1, 1, 0.2, 0.4, 0.4, 1.0]})
        self.N_states = len(S_df)

    def init_policy_rsa_psas(self):
        self.set_policy_rsa_psas()
        self.N_actions_in_s = []
        for s in range(self.N_states):
            self.N_actions_in_s.append(len(self.policy.a[self.policy.s == s]))

    def bellman_v(self, s: int = 0, forgetting_factor: float = 1.0) -> float:
        Gs = 0
        policy_s = self.policy[self.policy.s == s]
        for a in set(policy_s.a):
            policy = policy_s.pi[policy_s.a == a].to_numpy()[0]
            Gs += policy * self.bellman_q_by_v(
                s=s, a=a, forgetting_factor=forgetting_factor)
        # print('Gs =', Gs)
        # print('policy =', policy)
        return Gs

    def bellman_q_by_v(self, s: int = 0, a: int = 0, forgetting_factor: float = 1.0) -> float:
        Rsa = self.Rsa
        Psas = self.Psas
        reward = Rsa.R[(Rsa['s'] == s) & (Rsa['a'] == a)].to_numpy()[0]
        v_next = 0
        Psas_sa = Psas[(Psas.s == s) & (Psas.a == a)]
        for next_s in set(Psas_sa.next_s):
            if len(Psas_sa.P[Psas_sa.next_s == next_s]) and next_s < self.N_states - 1:
                # print(Psas_sa.P[Psas_sa.next_s==next_s],Psas_sa.P[Psas_sa.next_s==next_s].to_numpy()[0])
                v_next += Psas_sa.P[Psas_sa.next_s == next_s].to_numpy()[0] * self.v[next_s]
        Gs = reward + forgetting_factor * v_next
        return Gs

    def bellman_q(self, s: int = 0, a: int = 0, forgetting_factor: float = 1.0) -> float:
        Rsa = self.Rsa
        Psas = self.Psas
        reward = Rsa.R[(Rsa['s'] == s) & (Rsa['a'] == a)].to_numpy()[0]
        v_next = 0
        Psas_sa = Psas[(Psas.s == s) & (Psas.a == a)]
        for next_s in set(Psas_sa.next_s):
            if len(Psas_sa.P[Psas_sa.next_s == next_s]) and next_s < self.N_states - 1:
                v = 0
                policy_s = self.policy[self.policy.s == next_s]
                for next_a in set(policy_s.a):
                    policy = policy_s.pi[policy_s.a == next_a].to_numpy()[0]
                    v += policy * self.q[next_s][next_a]
                v_next += Psas_sa.P[Psas_sa.next_s == next_s].to_numpy()[0] * v
        Gs = reward + forgetting_factor * v_next
        return Gs


class MDP_PD_TTT(MDP_PD):
    def set_policy_rsa_psas(self):
        S_df = pd.DataFrame({'S': [0, 1, 2, 3, 4]})
        self.policy = pd.DataFrame(
            {'s': [0, 0, 0, 1, 2, 3], 'a': [0, 1, 2, 0, 0, 0], 'pi': [1 / 3, 1 / 3, 1 / 3, 1, 1, 1]})
        self.Rsa = pd.DataFrame({'s': [0, 0, 0, 1, 2, 3], 'a': [0, 1, 2, 0, 0, 0], 'R': [1, 0, -1 / 2, 0, 0, 0]})
        self.Psas = pd.DataFrame(
            {'s': [0, 0, 0, 0, 0, 1, 2, 3], 'a': [0, 1, 1, 2, 2, 0, 0, 0], 'next_s': [4, 1, 2, 3, 4, 4, 4, 4],
             'P': [1, 1 / 2, 1 / 2, 1 / 2, 1 / 2, 0, 0, 0]})
        self.N_states = len(S_df)


if __name__ == '__main__':
    MDP().test()
    print('---------------------------------')
    MDP_PD().test()
    print('---------------------------------')
    MDP_PD_TTT().test()

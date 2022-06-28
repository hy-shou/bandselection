
import numpy as np
import time
import sys
import pandas as pd

class BandSelect():
    def __init__(self):
        data = pd.read_csv('../data/cm/prisma_ig.csv', sep=',', index_col=0)
        self.action_space = data.columns
        self.n_actions = len(self.action_space)
        self.K = 60
        self.STEP_COUNT = 0
        self.state_num = len(self.action_space)
        self.r = data.values
        # self.state_list = np.identity(self.state_num)
        self.state_list = data.columns

    def step(self,observation, action):
        """
                执行动作。
                :param state: 当前状态。
                :param action: 执行的动作。
                :return:
                """
        reward = self.r[0][action]
        observation[0][action] = 1
        next_state = observation
        done = False

        if self.STEP_COUNT == self.K:
            done = True

        return next_state, reward, done




if __name__ == '__main__':
    env = BandSelect()
    # env.after(100, update)
    # env.mainloop()
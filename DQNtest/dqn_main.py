import numpy as np
from bs_dqn_env import BandSelect
from RL_DQN_brain import DeepQNetwork
import json

import random


def train_all_history():
    step = 0
    find = 0
    for episode in range(1000):
        # initial observation
        print(episode)
        observation = np.float32(np.zeros((1, env.n_actions)))

        while True:

            # RL choose action based on observation
            action = RL.choose_action(observation)

            # RL take action and get next observation and reward
            observation_, reward, done = env.step(observation, action)
            invalid_set = np.squeeze(np.argwhere(observation_ > 0))
            # print(invalid_set.shape)
            RL.store_transition(observation, action, reward, observation_,done)

            if (step > 200) and (step % 5 == 0):
                RL.learn()

            # swap observation
            observation = observation_
            env.STEP_COUNT += 1
            # break while loop when end of this episode
            if done:
                break
            step += 1

        print(step)
        env.STEP_COUNT = 0
        if np.mean(RL.cost_his[-env.K:]) <= 0.0001 and episode > 9000:
            RL.save_model('../data/dqn/' + str(env.K) +'_dqn.ckpt')
            find = 1
            break
    # end of game
    if find == 0:
        RL.save_model('../data/dqn/' + str(env.K) +'_dqn.ckpt')
    print('game over')
    # RL.plot_cost()


def test_all_history():
    # _38_ - dqn.ckpt
    RL.load_model('../data/dqn/' + str(env.K) +'_dqn.ckpt')
    current_state = random.randint(0, env.n_actions - 1)
    with open('../data/dqn/' + str(env.K) +'_bands.txt', 'w') as f:
        vec = ''
        for i in range(200):
            selected_bands = []
            observation = np.float32(np.zeros((1, env.n_actions)))
            observation[0][i] = 1

            env.STEP_COUNT += 1
            while True:

                # RL choose action based on observation
                action = RL.predict(observation)
                selected_bands.append(action)
                # RL take action and get next observation and reward
                observation_, reward, done = env.step(observation, action)
                observation = observation_
                env.STEP_COUNT += 1
                # break while loop when end of this episode
                if done:
                    break
            env.STEP_COUNT = 0
            vec += str(sorted(selected_bands)) + '\n'
        f.write(vec)
    # print(selected_bands)
    return selected_bands

if __name__ == "__main__":
    # maze game
    env = BandSelect()

    RL = DeepQNetwork(n_actions = env.state_num, state_list=env.state_list )
    train_all_history()
    test_all_history()
from functools import total_ordering
import numpy as np
from torch import float32, optim
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter, writer
import random
from skimage.transform import resize
from skimage.color import rgb2gray
import pickle
import matplotlib.pyplot as plt
import os



class DQN_network(nn.Module):

    def __init__(self):

        super(DQN_network, self).__init__()

        self.model = nn.Sequential(
            # first layer inputsize = 250
            nn.Conv2d(4,32,(8,8),(4,4)),
            nn.ReLU(),
            # second layer inputsize = 250
            nn.Conv2d(32,64,(4,4),(2,2)),
            nn.ReLU(),
            # fourth layer inputsize = 62
            nn.Conv2d(64,64,(3,3)),
            nn.ReLU(),
            # FC layer input features =  64 * 30 * 30 = 576000
            nn.Flatten(),
            nn.Linear(3136,512),
            nn.ReLU(),
            nn.Linear(512,256),
            nn.ReLU(),
            nn.Linear(256,6),
        )
        
        

    def forward(self, x):
        
        x = self.model(x)

        return x



class DQN_agent():

    def __init__(self,parms):
        self.eval_net = DQN_network()
        self.eval_net.load_state_dict(torch.load(parms))

    def choose_action(self, state):

        action = 0
       
        action_value = self.eval_net.forward(state)
            
        action = torch.max(action_value, 1)[1].data.numpy()
        action = action[0]  # get the action index
       

        return action
       



def resize_frame(ob):
    return np.array(resize(rgb2gray(ob)[22:-22], (84, 84))) .flatten()

def reset_frame_buffer():
    frames_buffer.clear()
    for i in range(4):
        frames_buffer.append(empty_frame)
    return

def obs_to_state(ob):
    this_frame = resize_frame(ob)
    frames_buffer.pop(0) # oldest frame discarded
    frames_buffer.append(this_frame)

    return torch.FloatTensor(np.array(tuple(frames_buffer)).reshape([1, 4, 84, 84]))

# Making 4 frames buffer
temp_env = gym.make('DemonAttack-v0')
empty_frame = resize_frame(temp_env.reset())
temp_env.close()
frames_buffer = list()

# Game playing

def test_playing(parms,rounds,record_tag):

    agent = DQN_agent(parms)
    env = gym.make('DemonAttack-v0')
    writer = SummaryWriter("logs")


    print("The DQN agent started playing...")
    total_iteration = 0

    for i in range(rounds):

        done = False
        reset_frame_buffer()
        state = obs_to_state(env.reset()) 
        round_reward = 0
        lives = 0
        

        while not done:

            env.render()
            if total_iteration % 4 == 0:
                action = agent.choose_action(state)
                next_state, reward, done, info = env.step(action)
            
                # manipulating reward
                lives_delta = info['lives'] - lives 
                lives = info['lives']
                if(lives_delta < 0):
                    reward = -20
                else:
                    reward = 0 if reward == 0 else 15*reward
            
                reward = reward if not done else -30

                round_reward += reward
                state = obs_to_state(next_state)  

                # printing info
                action_n = ["Nothing","Fire","R","L","R and F","L and F"]
                print("Action : {}".format(action_n[action]))


            
            # if total_iteration % 100 == 0:   
            #     for i in range(4):
            #         plt.subplot(1,4,i+1)
            #         plt.imshow(torch.FloatTensor(np.array(tuple(frames_buffer)).reshape([1, 4, 84, 84])).squeeze(0).permute(1,2,0)[:,:,i])
            #     plt.show()
            
            total_iteration += 1

                
            
        # recording
        writer.add_scalar(record_tag,round_reward,global_step=i)

    writer.close()
        
        

test_playing("parms/e_net.pkl",500,"Test_500_Rounds")



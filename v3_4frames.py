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

# parameters
EPSILON = 1.0  # initial exporation probability
GAMMA = 0.99  # decay value
Learning_Rate = 0.0001
Memory_Capa = 100000
N_Iteration = 100
Batch_size = 32
EPISODES = 400
learning_frequency = 4

# random seeds
random.seed(0)
np.random.seed(0)

# initiallizing the virtual game env
# retriving data and parameter patterns from the env
env = gym.make('DemonAttack-v0')
height, width, channels = env.observation_space.shape
N_actions = env.action_space.n

writer = SummaryWriter("logs")


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
            nn.Linear(256,N_actions),
        )
        
        

    def forward(self, x):
        
        x = self.model(x)

        return x



class DQN_agent():

    def __init__(self):
        self.eval_net, self.target_net = DQN_network(), DQN_network()
        self.target_net.load_state_dict(self.eval_net.state_dict()) # copy parameters
        # state, action ,reward and next state
        self.memory = [None for i in range(Memory_Capa)]
        self.memory_counter = 0
        self.learn_counter = 0
        self.optimizer = optim.Adam(self.eval_net.parameters(), Learning_Rate)
        self.loss = nn.MSELoss()


    def load_parameters(self,e,t):
        self.eval_net.load_state_dict(e)
        self.target_net.load_state_dict(t)
        self.optimizer = optim.Adam(self.eval_net.parameters(), Learning_Rate)


    def store_trans(self, state, action, reward, next_state,done):


        index = self.memory_counter % Memory_Capa



        trans = (state, [action], [reward], next_state, done)
        self.memory[index] = trans
        self.memory_counter += 1


    def choose_action(self, state):

        Qarray = None
        if random.random() >= EPSILON:  # choose actions from network
            action_value = self.eval_net.forward(state)
            # get argmax action of q values
            Qarray = action_value.detach().numpy()
            action = torch.max(action_value, 1)[1].data.numpy()
            action = action[0]  # get the action index
        else: # choose actions from random exploration
            action = np.random.randint(0, N_actions)

        # return max Q aswell
          
        return action,Qarray

    def learn(self):

        # learn 100 times then the target network update
        if self.learn_counter % N_Iteration == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_counter += 1

        sample_index = np.random.choice(Memory_Capa, Batch_size)

        batch_memory = []
        for i in range(len(sample_index)):

            index = sample_index[i]
            while self.memory[index] == None:
                index = np.random.choice(Memory_Capa, 1)[0]

            batch_memory.append(self.memory[index])

     
        # state, action ,reward and next state and done
        
        

        batch_state = batch_memory[0][0]
        batch_action = []
        batch_reward = []
        batch_next_state = batch_memory[0][3]

        for i in range(len(batch_memory)):
            if i != 0: # avoiding cat the existing very first memory
                batch_state = torch.cat((batch_state,batch_memory[i][0]),dim=0)
                batch_next_state = torch.cat((batch_next_state,batch_memory[i][3]),dim=0)
            batch_action.append(batch_memory[i][1])
            batch_reward.append(batch_memory[i][2])
            


        batch_state = torch.FloatTensor(batch_state)
        batch_action = torch.LongTensor(batch_action)
        batch_reward = torch.FloatTensor(batch_reward)
        batch_next_state = torch.FloatTensor(batch_next_state)


        q_eval = self.eval_net(batch_state).gather(1, batch_action)
        q_next = self.target_net(batch_next_state).detach()
        q_target = batch_reward + GAMMA * q_next.max(1)[0].view(Batch_size, 1) # max Q value for next state


        for i in range(len(q_target)):
            if batch_memory[i][4] == True:
                q_target[i] = batch_reward[i]
        
        

        loss = self.loss(q_eval,q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


# training

def resize_frame(ob):
    return np.array(resize(rgb2gray(ob)[22:-22], (84, 84))) .flatten()

# Making 4 frames buffer
temp_env = gym.make('DemonAttack-v0')
empty_frame = resize_frame(temp_env.reset())
temp_env.close()
frames_buffer = list()

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


agent = DQN_agent()
print("The DQN is collecting experience...")
step_counter_list = []
total_iteration = 0
freeze_para = 0
for episode in range(EPISODES):
    reset_frame_buffer()
    state = obs_to_state(env.reset()) 
    step_counter = 0
    episode_reward = 0
    lives = 0
    while True:

        step_counter += 1
        env.render()

        if total_iteration % 4 == 0:    

            action,Qarray = agent.choose_action(state)
            next_state, reward, done, info = env.step(action)
            next_state = obs_to_state(next_state)
            # lives_delta = info['lives'] - lives 
            # lives = info['lives']

            # if(lives_delta < 0):
            #     reward = -3
            # else:
            #     reward = 0 if reward == 0 else reward
            
            reward = reward if not done else -20
            
            episode_reward += reward
            agent.store_trans(state, action, reward, next_state,done)

            if total_iteration % learning_frequency == 0 and freeze_para == 1: # learning
                agent.learn()

            if agent.memory_counter >= Memory_Capa:
                agent.memory_counter = 0
            
        if total_iteration == 10000:
            freeze_para = 1 # able to train
            EPSILON = 0.22

            # viewing data frames
            # if total_iteration % 100 == 0:   
            #     for i in range(4):
            #         plt.subplot(1,4,i+1)
            #         plt.imshow(torch.FloatTensor(np.array(tuple(frames_buffer)).reshape([1, 4, 84, 84])).squeeze(0).permute(1,2,0)[:,:,i])
            #     plt.show()    

    

        if done:
            print("\n****Episode {} Ends, the reward is {}\n".format(episode, round(episode_reward, 3)))
            writer.add_scalar("RL_with_gaps",episode_reward,global_step=total_iteration)
            break
        
       

        state = next_state
        total_iteration += 1

        # output information
        if total_iteration % 100 == 0:
            print("-Episode  {} / {} iteration / Reward {} / Prob. Random Exp. {}".format(episode, total_iteration ,round(episode_reward, 3),EPSILON))
            action_n = ["Nothing","Fire","R","L","R and F","L and F"]
            print("Action : {}, Qarray: ".format(action_n[action]))
            print(Qarray)
            print("\n")

        # saving the network
        if total_iteration % 30000 == 0:
            
            if not os.path.exists("parms"):
                os.mkdir("parms")

            # network parameteres
            torch.save(agent.target_net.state_dict(),"parms/t_net.pkl")
            torch.save(agent.eval_net.state_dict(),"parms/e_net.pkl")

            


    # decent the greedy epsilon
    if EPSILON > 0.11:
        EPSILON -= 0.01

writer.close()





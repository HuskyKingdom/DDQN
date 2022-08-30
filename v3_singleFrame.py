import numpy as np
from torch import float32, optim
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter, writer
import random

# parameters
EPSILON = 1.0  # initial exporation probability
GAMMA = 0.99  # decay value
Learning_Rate = 0.0005
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
            nn.Conv2d(channels,32,(8,8),(4,4)),
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
        self.rt_memory_counter = 0
        self.learn_counter = 0
        self.optimizer = optim.Adam(self.eval_net.parameters(), Learning_Rate)
        self.loss = nn.MSELoss()

    def img_preprocessing(self,img):

        img = torch.FloatTensor(img).permute((2,0,1)) # reorder the channel 
        img = img.unsqueeze(0) # add another dimension
        return img # final shape -> (N,C,H,W)

    def store_trans(self, state, action, reward, next_state):

        if self.rt_memory_counter % 500 == 0:
            print("The experience pool collects {} time experience".format(self.rt_memory_counter))
        
        
        
        index = self.memory_counter % Memory_Capa



        trans = (state, [action], [reward], next_state)
        self.memory[index] = trans
        self.memory_counter += 1
        self.rt_memory_counter += 1

    def choose_action(self, state):


        if random.random() >= EPSILON:  # choose actions from network
            action_value = self.eval_net.forward(self.img_preprocessing(state))
            # get argmax action of q values
            action = torch.max(action_value, 1)[1].data.numpy()
            action = action[0]  # get the action index
        else: # choose actions from random exploration
            action = np.random.randint(0, N_actions)
        return action

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

     
        # state, action ,reward and next state
        

        batch_state = []
        batch_action = []
        batch_reward = []
        batch_next_state = []

        for i in range(len(batch_memory)):
            batch_state.append(batch_memory[i][0])
            batch_action.append(batch_memory[i][1])
            batch_reward.append(batch_memory[i][2])
            batch_next_state.append(batch_memory[i][3])


        batch_state = torch.FloatTensor(batch_state)
        batch_action = torch.LongTensor(batch_action)
        batch_reward = torch.FloatTensor(batch_reward)
        batch_next_state = torch.FloatTensor(np.array(batch_next_state)).permute(0,3,1,2)

        q_eval = self.eval_net(batch_state).gather(1, batch_action)
        q_next = self.target_net(batch_next_state).detach()
        q_target = batch_reward + GAMMA * q_next.max(1)[0].view(Batch_size, 1) # max Q value for next state

        loss = self.loss(q_eval, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()



agent = DQN_agent()
print("The DQN is collecting experience...")
step_counter_list = []
total_iteration = 0
freeze_para = 0
for episode in range(EPISODES):
    state = env.reset()
    step_counter = 0
    episode_reward = 0
    lives = 0
    while True:
        step_counter += 1
        env.render()
        action = agent.choose_action(state)
        next_state, reward, done, info = env.step(action)
        
        lives_delta = lives - info['lives']
        lives = info['lives']

        if(lives_delta < 0):
            reward = -30
        else:
            reward = -1 if reward == 0 else 10*reward
        
        reward = reward if not done else -20
        
        episode_reward += reward
        agent.store_trans(state, action, reward, next_state)

        if total_iteration % learning_frequency == 0 and freeze_para == 1: # learning
            agent.learn()

        if agent.memory_counter >= Memory_Capa:
            agent.memory_counter = 0
        
        if total_iteration == 20000:
            freeze_para = 1 # able to train
            EPSILON = 0.28

        if done:
            print("episode {}, the reward is {}".format(episode, round(episode_reward, 3)))
            writer.add_scalar("RL_with_different_reward",episode_reward,global_step=total_iteration)
            break
        
       

        state = next_state
        total_iteration += 1
    
    # decent the greedy epsilon
    if EPSILON > 0.15:
        EPSILON -= 0.01
        print("Chance of exploration reduced to {}",format(EPSILON))

writer.close()





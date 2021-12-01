'''
    1. Don't delete anything which is already there in code.
    2. you can create your helper functions to solve the task and call them.
    3. Don't change the name of already existing functions.
    4. Don't change the argument of any function.
    5. Don't import any other python modules.
    6. Find in-line function comments.

'''

import gym
import numpy as np
import math
import time
import argparse
import matplotlib.pyplot as plt
import random


class sarsaAgent():
    '''
    - constructor: graded
    - Don't change the argument of constructor.
    - You need to initialize epsilon_T1, epsilon_T2, learning_rate_T1, learning_rate_T2 and weight_T1, weights_T2 for task-1 and task-2 respectively.
    - Use constant values for epsilon_T1, epsilon_T2, learning_rate_T1, learning_rate_T2.
    - You can add more instance variable if you feel like.
    - upper bound and lower bound are for the state (position, velocity).
    - Don't change the number of training and testing episodes.
    '''

    def __init__(self):
        self.env = gym.make('MountainCar-v0')
        self.epsilon_T1 = 0.05
        self.epsilon_T2 = 0.05
        self.learning_rate_T1 = 0.1
        self.learning_rate_T2 = 0.1
        self.weights_T1 = np.random.randn(40,40,self.env.action_space.n) #None
        self.num_tiles_T2 = 10
        self.discrete_poses_T2 = 40
        self.discrete_vels_T2 = 40
        self.weights_T2 = np.random.randn(40,40,self.env.action_space.n)
        self.discount = 1.0
        self.train_num_episodes = 10000
        self.test_num_episodes = 100
        self.upper_bounds = [self.env.observation_space.high[0], self.env.observation_space.high[1]]
        self.lower_bounds = [self.env.observation_space.low[0], self.env.observation_space.low[1]]

    '''
    - get_table_features: Graded
    - Use this function to solve the Task-1
    - It should return representation of state.
    '''

    def get_table_features(self, obs):
    	pos, vel = obs
    	table_pos = np.linspace(self.lower_bounds[0],self.upper_bounds[0], 40)
    	table_vel = np.linspace(self.lower_bounds[1], self.upper_bounds[1] ,40)
    	tabular_pos = np.digitize(pos, table_pos)
    	tabular_vel = np.digitize(vel, table_vel)
    	states = [tabular_pos,tabular_vel]
    	return states

    '''
    - get_better_features: Graded
    - Use this function to solve the Task-2
    - It should return representation of state.
    '''
    
    def get_better_features(self, obs):
    	pos, vel = obs
    	table_pos = np.linspace(self.lower_bounds[0],self.upper_bounds[0], 40)
    	table_vel = np.linspace(self.lower_bounds[1], self.upper_bounds[1] ,40)
    	tabular_pos = np.digitize(pos, table_pos)
    	tabular_vel = np.digitize(vel, table_vel)
    	states = [tabular_pos,tabular_vel]
    	return states

    '''
    - choose_action: Graded.
    - Implement this function in such a way that it will be common for both task-1 and task-2.
    - This function should return a valid action.
    - state representation, weights, epsilon are set according to the task. you need not worry about that.
    '''
    
    def choose_action(self,state,weights,epsilon):
        #print(state)
        if np.random.uniform(0,1) <epsilon:
            action = self.env.action_space.sample()
            #print("yes")
        else:
            #print(state)
            action = np.argmax(weights[state[0],state[1],:])
            #print(weights.shape)
            #print(weights[state,:].shape)
            #print(action)
        return action
    
    

    '''
    - sarsa_update: Graded.
    - Implement this function in such a way that it will be common for both task-1 and task-2.
    - This function will return the updated weights.
    - use sarsa(0) update as taught in class.
    - state representation, new state representation, weights, learning rate are set according to the task i.e. task-1 or task-2.
    '''
    
    def sarsa_update(self, state, action, reward, new_state, new_action, learning_rate, weights):
  
        if task =="T2":
            
            pl = max(state[0]-2,0)
            pr = min(state[0]+3,40)
            vl = max(state[1]-2,0)
            vr = min(state[1]+3,40)
            
            rbf = np.zeros((pr-pl,vr-vl))
            for i in range(pr-pl):
                for j in range(vr-vl):
                    rbf[i,j] = 1/(1+abs(i-2)+abs(j-2))
            
            pln = max(new_state[0]-2,0)
            prn = min(new_state[0]+3,40)
            vln = max(new_state[1]-2,0)
            vrn = min(new_state[1]+3,40)
            
            difp = abs(pr-pl - (prn-pln))
            difv = abs(vr-vl - (vrn-vln))
            
            new = weights[pln:prn,vln:vrn,action]
            new = np.pad(new, [(0, difp), (0, difv)], mode='constant')
            new = new[0:pr-pl,0:vr-vl]
            target = reward + new
            weights[pl:pr,vl:vr,action] = weights[pl:pr,vl:vr,action] + learning_rate*rbf*(target -weights[pl:pr,vl:vr,action])
        else:
            target = reward + weights[new_state[0],new_state[1],new_action]
            weights[state[0],state[1],action] = weights[state[0],state[1],action] + learning_rate*(target - weights[state[0],state[1],action])
        return weights
    
    

    '''
    - train: Ungraded.
    - Don't change anything in this function.
    
    '''

    def train(self, task='T1'):
        if (task == 'T1'):
            get_features = self.get_table_features
            weights = self.weights_T1
            epsilon = self.epsilon_T1
            learning_rate = self.learning_rate_T1
        else:
            get_features = self.get_better_features
            weights = self.weights_T2
            epsilon = self.epsilon_T2
            learning_rate = self.learning_rate_T2
        reward_list = []
        plt.clf()
        plt.cla()
        for e in range(self.train_num_episodes):
            current_state = get_features(self.env.reset())
            done = False
            t = 0
            new_action = self.choose_action(current_state, weights, epsilon)
            while not done:
                action = new_action
                obs, reward, done, _ = self.env.step(action)
                new_state = get_features(obs)
                new_action = self.choose_action(new_state, weights, epsilon)
                weights = self.sarsa_update(current_state, action, reward, new_state, new_action, learning_rate,
                                            weights)
                current_state = new_state
                if done:
                    reward_list.append(-t)
                    #if e%500==0:
                    	#print(-t)
                    break
                t += 1
        self.save_data(task)
        reward_list=[np.mean(reward_list[i-100:i]) for i in range(100,len(reward_list))]
        plt.plot(reward_list)
        plt.savefig(task + '.jpg')

    '''
       - load_data: Ungraded.
       - Don't change anything in this function.
    '''

    def load_data(self, task):
        return np.load(task + '.npy')

    '''
       - save_data: Ungraded.
       - Don't change anything in this function.
    '''

    def save_data(self, task):
        if (task == 'T1'):
            with open(task + '.npy', 'wb') as f:
                np.save(f, self.weights_T1)
            f.close()
        else:
            with open(task + '.npy', 'wb') as f:
                np.save(f, self.weights_T2)
            f.close()

    '''
    - test: Ungraded.
    - Don't change anything in this function.
    '''

    def test(self, task='T1'):
        if (task == 'T1'):
            get_features = self.get_table_features
        else:
            get_features = self.get_better_features
        weights = self.load_data(task)
        reward_list = []
        for e in range(self.test_num_episodes):
            current_state = get_features(self.env.reset())
            done = False
            t = 0
            while not done:
                action = self.choose_action(current_state, weights, 0)
                obs, reward, done, _ = self.env.step(action)
                new_state = get_features(obs)
                current_state = new_state
                if done:
                    reward_list.append(-1.0 * t)
                    break
                t += 1
        return float(np.mean(reward_list))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True,
       help="first operand", choices={"T1", "T2"})
    ap.add_argument("--train", required=True,
       help="second operand", choices={"0", "1"})
    args = vars(ap.parse_args())
    task=args['task']
    train=int(args['train'])
    agent = sarsaAgent()
    agent.env.seed(0)
    np.random.seed(0)
    agent.env.action_space.seed(0)
    if(train):
        agent.train(task)
    else:
        print(agent.test(task))

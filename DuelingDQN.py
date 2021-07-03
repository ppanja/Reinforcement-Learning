import gym
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.python.keras.layers import Dense
from tensorflow.keras.optimizers import Adam


# Implementation of replaybuffer
class ReplayBuffer:
    
    def __init__(self, size, input_shape):
        self.size = size
        self.counter = 0
        self.state_buffer = np.zeros((self.size, input_shape), dtype=float)
        self.action_buffer = np.zeros(self.size, dtype=int)
        self.reward_buffer = np.zeros(self.size, dtype=float)
        self.next_state_buffer = np.zeros((self.size, input_shape), dtype=float)
        self.terminal_buffer = np.zeros(self.size, dtype=bool)

    
    def store_tuples(self, state, action, reward, next_state, done):
        i = self.counter % self.size
        self.state_buffer[i] = state
        self.action_buffer[i] = action
        self.reward_buffer[i] = reward
        self.next_state_buffer[i] = next_state
        self.terminal_buffer[i] = done
        self.counter += 1

    
    def sample_buffer(self, batch_size):
        max_buffer = min(self.counter, self.size)
        batch = np.random.choice(max_buffer, batch_size, replace=False)
        state_batch = self.state_buffer[batch]
        action_batch = self.action_buffer[batch]
        reward_batch = self.reward_buffer[batch]
        next_state_batch = self.next_state_buffer[batch]
        done_batch = self.terminal_buffer[batch]

        return state_batch, action_batch, reward_batch, next_state_batch, done_batch

# Implementation of network
class DuelingDQN(keras.Model):
    
    def __init__(self, num_actions, fc1, fc2): # Initialise the network
        super(DuelingDQN, self).__init__()
        self.dense1 = Dense(fc1, activation='relu')
        self.dense2 = Dense(fc2, activation='relu')
        self.V = Dense(1, activation=None)
        self.A = Dense(num_actions, activation=None)

    def call(self, state):
        x = self.dense1(state)
        x = self.dense2(x)
        V = self.V(x)
        A = self.A(x)
        avg_A = tf.math.reduce_mean(A, axis=1, keepdims=True)
        Q = (V + (A - avg_A))

        return Q, A
		
        
class Agent:
    def __init__(self, lr, gamma, epsilon, batch_size):
        input_dim = 8
        num_actions = 4
        self.action_space = [i for i in range(num_actions)]
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.epsilon_decay = 0.9995
        self.epsilon_min = 0.1
        self.update_rate = 120
        self.step_counter = 0
        self.buffer = ReplayBuffer(500000, input_dim)
        
        self.model = DuelingDQN(num_actions, 512, 256)
        self.target_model = DuelingDQN(num_actions, 512, 256)
        self.model.compile(optimizer=Adam(learning_rate=lr), loss='mse')
        self.target_model.compile(optimizer=Adam(learning_rate=lr), loss='mse')

    def store_tuple(self, state, action, reward, new_state, done):
        self.buffer.store_tuples(state, action, reward, new_state, done)

    def get_action(self, observation):
        if np.random.random() < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            state = np.array([observation])
            _, actions = self.model(state)
            action = tf.math.argmax(actions, axis=1).numpy()[0]

        return action
    
    # TRAIN FROM REPLAYBUFFER
    def train(self):
        if self.buffer.counter < self.batch_size:
            return
        if self.step_counter % self.update_rate == 0:
            self.target_model.set_weights(self.model.get_weights())

        state_batch, action_batch, reward_batch, new_state_batch, done_batch = \
            self.buffer.sample_buffer(self.batch_size)

        q_predicted, _ = self.model(state_batch)
        q_next, _ = self.target_model(new_state_batch)
        q_max_next = tf.math.reduce_max(q_next, axis=1, keepdims=True).numpy()
        q_target = np.copy(q_predicted)

        for idx in range(done_batch.shape[0]):
            target_q_val = reward_batch[idx]
            if not done_batch[idx]:
                target_q_val += self.gamma*q_max_next[idx]
            q_target[idx, action_batch[idx]] = target_q_val
        self.model.train_on_batch(state_batch, q_target)
        self.step_counter += 1

    # TRAIN THE DUELINGDQN MODEL
    def train_model(self, env, num_episodes, earlystopping=True):

        scores, episodes, avg_scores, obj = [], [], [], []
        goal = 150
        f = 0
        avg_score = 0
        
        t1 = time.perf_counter()

        for i in range(num_episodes):
            
            # Early stopping when targeted reward goal is reached
            if earlystopping:
                if avg_score > goal:
                    # Save model and weights, and return
                    self.model.save(("saved_networks/duelingdqn_model_{0}_{1}_{2}".format(i, self.lr, self.gamma)))
                    self.model.save_weights(("saved_networks/duelingdqn_model_{0}_{1}_{2}/net_weights_{0}_{1}_{2}.h5".format(i, self.lr, self.gamma)))
                    print("Saved trained model and weights")
                    
                    return scores, avg_scores

            done = False
            score = 0.0
            state = env.reset()
            while not done:
                action = self.get_action(state)
                new_state, reward, done, _ = env.step(action)
                score += reward
                self.store_tuple(state, action, reward, new_state, done)
                state = new_state
                self.train()
            scores.append(score)
            obj.append(goal)
            episodes.append(i)
            avg_score = np.mean(scores[-100:])
            avg_scores.append(avg_score)

            # print every print_count
            print_count = 50
            if (i % print_count == 0) and (i != 0):
                print("Episode {}/{}, Score: {} epsilon: {}, AVG Score: {}".format(i, num_episodes, score, round(self.epsilon, 2), round(avg_score, 2)))
                t2 = time.perf_counter()
                print("lr={} gamma={} Finished {} episodes in {} seconds. Running...".format(self.lr, self.gamma, print_count, t2-t1))
                t1 = time.perf_counter()
    
            if self.epsilon > self.epsilon_min: # epsilon decay
                self.epsilon *= self.epsilon_decay
        
           # Save the model and weights before and after the train
            if i==num_episodes-1:  # Save the model in the last episode
                
                self.model.save(("saved_networks/duelingdqn_model_{0}_{1}_{2}".format(i, self.lr, self.gamma)))
                self.model.save_weights(("saved_networks/duelingdqn_model_{0}_{1}_{2}/net_weights_{0}_{1}_{2}.h5".format(i, self.lr, self.gamma)))
                print("Saved trained model")

        return scores, avg_scores
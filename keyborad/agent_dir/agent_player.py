from agent_dir.agent import Agent
import numpy as np
import sys
import os
import re
import math
import random
import time
from pickle import dump, load
from datetime import datetime

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
np.set_printoptions(threshold=np.inf)

class Agent_Player(Agent):

    def key_ctrl(self, key, flag):
        #key list:
        #: 00: noop
        #: 01: fire
        #: 02: up
        #: 03: left
        #: 04: right
        #: 05: down
        #: 06: right up
        #: 07: left down
        #: 08: right down
        #: 09: left up
        #: 10: up fire
        #: 11: left fire
        #: 12: right fire
        #: 13: down fire
        #: 14: right up fire
        #: 15: left down fire
        #: 16: right down fire
        #: 17: left up fire
        
        ACTION_U = 2 
        ACTION_R = 3 
        ACTION_L = 4
        ACTION_D = 5
        ACTION_RU = 6 
        ACTION_LD = 7 
        ACTION_RD = 8
        ACTION_LU = 9
        
        if key == 0xff52: self.press_up = flag
        if key == 0xff51: self.press_left = flag
        if key == 0xff53: self.press_right = flag
        if key == 0xff54: self.press_down = flag
        if key == 122: self.press_fire = flag

        # detect horizontal & verticle movement
        # axis_h set to 0 if left & right both pressed or both not pressed
        # axis_v set to 0 if up & down both pressed or both not pressed
         
        axis_h = 0
        axis_v = 0      
        if self.press_up: axis_v += 1
        if self.press_down: axis_v -= 1
        if self.press_left: axis_h -= 1
        if self.press_right: axis_h += 1
        
        action = 0
        if axis_h == 0 and axis_v == 1:
            action = ACTION_U
        elif axis_h == 0 and axis_v == -1:
            action = ACTION_D
        elif axis_h == -1 and axis_v == 0:
            action = ACTION_L
        elif axis_h == 1 and axis_v == 0:
            action = ACTION_R
        elif axis_h == 1 and axis_v == 1:
            action = ACTION_RU
        elif axis_h == 1 and axis_v == -1:
            action = ACTION_RD
        elif axis_h == -1 and axis_v == 1:
            action = ACTION_LD
        elif axis_h == -1 and axis_v == -1:
            action = ACTION_LU
        
        # +8 makes player fire while moving
        if self.press_fire:
            if action == 0:
                action = 1
            else:
                action += 8
        if action not in self.env.unwrapped._action_set:
            print('{} is not in action space!'.format(action))
            action = 0
        action = np.where(self.env.unwrapped._action_set == action)[0][0]
        self.human_agent_action = action

    def key_press(self, key, mod):
        if key==0xff1b: self.human_wants_quit = True
        if key==0xff0d: self.human_wants_restart = True
        if key==32: self.human_sets_pause = not self.human_sets_pause

        self.key_ctrl(key, True)

    def key_release(self, key, mod):
        self.key_ctrl(key, False)

    def __init__(self, env, args):
        """
        Initialize every things you need here.
        For example: building your model
        """
        np.random.seed(1)

        super(Agent_Player,self).__init__(env)
        
        self.env.render()
        self.env.unwrapped.viewer.window.on_key_press = self.key_press
        self.env.unwrapped.viewer.window.on_key_release = self.key_release
        print(env.unwrapped.get_action_meanings())
        print(self.env.unwrapped._action_set)
        
        self.human_agent_action = 0
        self.human_wants_restart = False
        self.human_wants_quit = False
        self.human_sets_pause = False

        self.press_fire = False
        self.press_left = False
        self.press_right = False
        self.press_up = False
        self.press_down  = False

        self.display = 200
        self.iteration = 30000
        self.batch_size = 32
        self.learning_rate = 0.0002
        
        self.n_action = env.action_space.n
        self.n_input = env.observation_space.low.size
        self.memory = []
        
    def init_game_setting(self):
        """

        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary

        """
        ##################
        # YOUR CODE HERE #
        ##################
        pass


    def save_memory(self, s, a, r, d, s_):
        done = float(d)
        transition = np.hstack((s, [a, r, done], s_))
        i = self.memory_count % self.memory_size
        self.memory[i, :] = transition
        self.memory_count += 1


    def train(self):
    
        obs = self.env.reset()
#        obs = np.reshape(obs, [self.n_input])
        
        step = 0
        score = 0
        while(True):
            if self.human_wants_quit:
                return
           
            action = self.make_action(obs, test=False)
            obs_, reward, done, info = self.env.step(action)
#            obs_ = np.reshape(obs_, [self.n_input])
#            self.save_memory(obs, action, reward, done, obs_)
            self.memory.append([obs, action, obs_, reward, done])
            # step forward
            time.sleep(0.05)
            obs = obs_
            step += 1
            score += reward
            while self.human_sets_pause:
                self.env.render()
                time.sleep(0.1)
            
#            transition = np.hstack((obs, action, reward, obs_, d))  
            if done:
                date = datetime.now().strftime("%d-%H-%M")
                f = open('records/{}'.format(date), 'wb')
                dump(self.memory, f)
                f.close()
                print (step)
                print (score)
#               if step < self.memory_size:
#                   np.save('record_s%5d_r%5d.npy' % (step, score), self.memory[:step])
#               else:
#                   np.save('record_s%5d_r%5d.npy' % (step, score), self.memory)
#   
                break
    

    def make_action(self, observation, test=True):
        obs = np.reshape(observation, [-1, self.n_input])
        action = self.human_agent_action
        return action


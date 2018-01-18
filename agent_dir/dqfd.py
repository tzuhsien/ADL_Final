from agent_dir.agent import Agent

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F

import numpy as np

from collections import namedtuple
from itertools import count
import random
import os
import pickle
import tqdm
import datetime
import sys

use_cuda = torch.cuda.is_available()
Tensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class Agent_DQN(Agent):
    def __init__(self, env, args):
        """
        Initialize every things you need here.
        For example: building your model
        """

        super(Agent_DQN,self).__init__(env)
        
        # hyperparameter

        self.buffer_size = 10000
        self.start_step = 10000
        self.max_step = 10000000
        self.learning_rate = 1e-4
        self.update_target_step = 1000
        self.update_online_step = 4
        self.gamma = 0.99
        self.batch_size = 64
        self.num_actions = env.action_space.n

        self.pretrain_iter = 750000 #750000
        self.demo_props = 0.3
        self.n_step = 10
        self.n_step_weight = 1
        self.supervised_weight = 1
        self.margin = 0.8

        # set explore schedule

        self.schedule = ExplorationSchedule()

        # set replay buffer
        if not args.test:
            self.read_demo_file(args.demo_file)
        self.buffer = ReplayMemory(self.buffer_size)

        # build Q networks
        self.Q = DQN(self.num_actions)
        self.Q_target = DQN(self.num_actions)
        self.opt = optim.RMSprop(self.Q.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        self.loss_func = nn.MSELoss(size_average=False)
        
        self.record_dir = os.path.join('dqn_record', '{}_{}'.format(args.env_name, datetime.datetime.now().strftime("%m-%d_%H-%M")))
        
        if args.test:
            print('loading trained model')
            self.Q.load_state_dict(torch.load(os.path.join(args.model_path)))
            self.Q_target.load_state_dict(self.Q.state_dict())
        
            self.record_dir = os.path.join('dqn_record', '{}_{}_test'.format(args.env_name, datetime.datetime.now().strftime("%m-%d_%H-%M")))
 

        if not os.path.exists(self.record_dir):
            os.makedirs(self.record_dir)

        if use_cuda:
            self.Q = self.Q.cuda()
            self.Q_target = self.Q_target.cuda()
            self.loss_func = self.loss_func.cuda()
        # initialize

        self.time = 0

    def read_demo_file(self, filename):
        #state, action, next_state, reward, done, n_reward, n_next_state
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        
        self.demo_buffer = ReplayMemory(len(data))
        for transition in data:
            self.demo_buffer.push(transition)

    def init_game_setting(self):
        """

        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary

        """
        pass

    def optimize_model(self, demo_props):
        demo_samples = int(demo_props * self.batch_size)
        b_demo_memory = self.demo_buffer.sample(demo_samples)
        if len(self.buffer) < self.batch_size - demo_samples:
            return
        b_memory = self.buffer.sample(self.batch_size - demo_samples)
        b_memory = b_demo_memory + b_memory

        b_s, b_a, b_r, b_s_, b_nd, b_nr, b_ns, b_nns = [], [], [], [], [], [], [], []
        for s, a, s_, r, d, nr, ns in b_memory:
            b_s.append(s)
            b_a.append(a)
            b_r.append(r)
            b_s_.append(s_)
            b_nd.append(not d)

            b_nr.append(nr)

            if ns is not None:
                b_ns.append(ns)
                b_nns.append(1)
            else:
                b_ns.append(s_)
                b_nns.append(0)
        # print(b_nd)

        b_s = Variable(torch.FloatTensor(np.array(b_s)).transpose(1, 3))
        b_a = Variable(torch.LongTensor(np.array(b_a)).unsqueeze(1))
        b_r = Variable(torch.FloatTensor(np.array(b_r)).unsqueeze(1))
        b_s_ = Variable(torch.FloatTensor(np.array(b_s_)).transpose(1, 3))
        b_nd = Variable(torch.Tensor(b_nd).unsqueeze(1))

        b_nr = Variable(torch.Tensor(np.array(b_nr)).unsqueeze(1))
        b_ns = Variable(torch.Tensor(np.array(b_ns)).transpose(1, 3))
        b_nns = Variable(torch.Tensor(b_nns).unsqueeze(1))
        #print(b_nd.size(), b_nns.size())

        if use_cuda:
            b_s = b_s.cuda()
            b_a = b_a.cuda()
            b_r = b_r.cuda()
            b_s_ = b_s_.cuda()
            b_nd = b_nd.cuda()
            b_nr = b_nr.cuda()
            b_ns = b_ns.cuda()
            b_nns = b_nns.cuda()

        q_vals = self.Q(b_s)
        q_eval = q_vals.gather(1, b_a)

        # next step
        q_next = self.Q_target(b_s_).detach()
        q_next.volatile = False
        values, indices = torch.max(self.Q(b_s_), 1)
        q_true = q_next.gather(1, indices.unsqueeze(1))

        # n steps
        q_n_next = self.Q_target(b_ns).detach()
        q_n_next.volatile = False
        values, indices = torch.max(self.Q(b_ns), 1)
        q_n_true = q_n_next.gather(1, indices.unsqueeze(1))
        
        # q_target = b_r + self.gamma * q_next.max(1, keepdim=True)[0] * b_nd
        q_target = b_r + self.gamma * q_true * b_nd
        q_n_reward = b_nr + pow(self.gamma, self.n_step) * q_n_true * b_nns

        q_loss = self.loss_func(q_eval, q_target)
        n_step_loss = self.loss_func(q_eval, q_n_reward)

        # supervised loss
        supervised_loss = Variable(torch.zeros(1)).cuda()
        if demo_samples:
            margins = (torch.ones(self.num_actions, self.num_actions) - torch.eye(self.num_actions)) * self.margin
            batch_margins = margins[b_a.data.squeeze().cpu()]
            q_vals += Variable(batch_margins).cuda()
            supervised_loss = (q_vals.max(1)[0].unsqueeze(1) - q_eval)[:demo_samples].sum()

        loss = q_loss + self.n_step_weight * n_step_loss + self.supervised_weight * supervised_loss
        #print('{:10.3f} {:10.3f} {:10.3f} {:10.3f}'.format(loss.data[0], q_loss.data[0], n_step_loss.data[0], supervised_loss.data[0]), file=sys.stderr) 
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

    def train(self):
        """
        Implement your training algorithm here
        """

        print('start to pretrain')
        for idx in range(self.pretrain_iter):
            self.optimize_model(1.0)

            if idx % self.update_target_step == 0:
                self.Q_target.load_state_dict(self.Q.state_dict())

            if idx % 10000 == 10000-1:
                state = self.env.reset()
                done = False
                eps_reward = 0
                cnt = 10
                for _ in range(cnt):
                    while not done:
                        action = self.make_action(state, test=False)
                        _, reward, done, _ = self.env.step(action)
                        eps_reward += reward
                print('pretraining: {} steps -> {}'.format(idx, eps_reward/cnt))
        print('finish pretraining')
        
        self.rewards = []
        for num_episode in count(1):
            state = self.env.reset()
            done = False
            self.eps_reward = 0

            local_buffer = []
            while not done:
                action = self.make_action(state, test=False)
                next_state, reward, done, _ = self.env.step(action)
                self.eps_reward += reward
                self.time += 1

                local_buffer.insert(0, [state, action, next_state, reward, done, 0, None])

                gamma = 1
                for idx in range(len(local_buffer)):
                    local_buffer[idx][5] += gamma * reward
                    gamma *= self.gamma

                if len(local_buffer) == self.n_step:
                    last_trans = local_buffer.pop()
                    if not done:
                        last_trans[6] = next_state
                    self.buffer.push(last_trans)

                state = next_state

                if self.time % self.update_online_step == 0:
                    eps = self.schedule.value(self.time)
                    if len(self.buffer) == self.buffer.capacity:
                        self.optimize_model(1 - eps)

                if self.time % self.update_target_step == 0:
                    self.Q_target.load_state_dict(self.Q.state_dict())
                    torch.save(self.Q.state_dict(), os.path.join(self.record_dir, 'dqfd_DQN.pkl'))

                if self.time % 1000 == 0:
                    print('Now playing %d steps.' % (self.time))

            for trans in local_buffer:
                self.buffer.push(trans)

            print('Step: %d, Episode: %d, Episode Reward: %f' % (self.time, num_episode, self.eps_reward))

            with open(os.path.join(self.record_dir, 'dqfd_DQN.csv'), 'a') as f:
                print("{}, {}".format(self.time, self.eps_reward), file=f)
            
            self.rewards.append(self.eps_reward)
            if num_episode % 100 == 0:
                print('---')
                print('Recent 100 episode: %f' % (sum(self.rewards) / 100))
                print('---')
                self.rewards = []

            if self.time >= 7000000:
                break
            


    def make_action(self, observation, test=True):
        """
        Return predicted action of your agent

        Input:
            observation: np.array
                stack 4 last preprocessed frames, shape: (84, 84, 4)

        Return:
            action: int
                the predicted action from trained model
        """
        obs = Variable(torch.from_numpy(observation).unsqueeze(0).transpose(1, 3))
        if use_cuda:
            obs = obs.cuda()

        if not test:
            rd = random.random()
            if rd < 0.95:
                return self.Q(obs).data.max(1)[1][0]
            else:
                return self.env.get_random_action()
        else:
            return self.Q(obs).data.max(1)[1][0]

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, transition):
        if self.position < self.capacity:
            self.memory.append(None)
        index = self.position % self.capacity
        self.memory[index] = transition
        self.position += 1

    def sample(self, batch_size=32):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):
    def __init__(self, num_actions):

        super(DQN, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=4, 
                      out_channels=32, 
                      kernel_size=8, 
                      stride=4),
            nn.ReLU(),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, 
                      out_channels=64, 
                      kernel_size=4, 
                      stride=2),
            nn.ReLU(),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=64, 
                      out_channels=64, 
                      kernel_size=3, 
                      stride=1),
            nn.ReLU(),
        )

        layer4 = nn.Sequential()
        layer4.add_module('fc1', nn.Linear(64 * 7 * 7, 512))
        layer4.add_module('lrelu1', nn.LeakyReLU())
        layer4.add_module('fc2', nn.Linear(512, num_actions))
        self.layer4 = layer4

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        fc_input = x.view(x.size(0), -1)
        fc_out = self.layer4(fc_input)
        return fc_out

class ExplorationSchedule(object):
    def __init__(self, timestep=1e6, final=0.95, initial=0.5):
        self.timestep = timestep
        self.final = final
        self.initial = initial

    def value(self, t):
        return self.initial + (self.final - self.initial) * min(t / self.timestep, 1.0)

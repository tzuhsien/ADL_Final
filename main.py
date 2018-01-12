"""

### NOTICE ###
You DO NOT need to upload this file

"""
import argparse
from test import test
from environment import Environment


def parse():
    parser = argparse.ArgumentParser(description="MLDS&ADL Final Project", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env_name', default='SeaquestNoFrameskip-v4', help='environment name - SeaquestNoFrameskip-v4, EnduroNoFrameskip-v4, SpaceInvadersNoFrameskip-v4')
    parser.add_argument('--train', action='store_true', help='whether train or not')
    parser.add_argument('--test', action='store_true', help='whether test or not')
    parser.add_argument('--type', default='human', help='human, double, dueling, dqfd')
    parser.add_argument('--video_dir', default=None, help='output video directory')
    parser.add_argument('--do_render', action='store_true', help='whether render environment')
    try:
        from argument import add_arguments
        parser = add_arguments(parser)
    except:
        pass
    args = parser.parse_args()
    return args


def run(args):

    env = Environment(args.env_name, args, atari_wrapper=True)
    if args.type == 'human':
        from agent_dir.agent_player import Agent_Player
        agent = Agent_Player(env, args)
    elif args.type == 'double':
        from agent_dir.double_dqn import Agent_DQN
        agent = Agent_DQN(env, args)
    elif args.type == 'dueling':
        from agent_dir.duel_network import Agent_DQN
        agent = Agent_DQN(env, args)
    elif args.type == 'dqfd':
        print("Not yet implement!")
        exit()
    else:
        print("Unknown type: {}".format(args.type))
        exit()

    if args.train:
        agent.train()

    if args.test:
        test(agent, env, total_episodes=100)


if __name__ == '__main__':
    args = parse()
    run(args)

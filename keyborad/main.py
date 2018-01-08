"""

### NOTICE ###
You DO NOT need to upload this file

"""
import argparse
from test import test
from environment import Environment


def parse():
    parser = argparse.ArgumentParser(description="MLDS&ADL HW3")
    parser.add_argument('--env_name', default=None, help='environment name')
    parser.add_argument('--train', action='store_true', help='whether train policy gradient')
    parser.add_argument('--test', action='store_true', help='whether test policy gradient')
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
    if args.train:
        env_name = args.env_name or 'Seaquest-v0'
        env = Environment(env_name, args, atari_wrapper=True)
        from agent_dir.agent_player import Agent_Player
        agent = Agent_Player(env, args)
        agent.train()

    if args.test:
        env = Environment('Seaquest-v0', args, atari_wrapper=True, test=True)
        from agent_dir.agent_player import Agent_Player
        agent = Agent_Player(env, args)
        test(agent, env, total_episodes=100)


if __name__ == '__main__':
    args = parse()
    run(args)

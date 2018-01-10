import numpy as np
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--file_list", type=str, default=None, help="file list")
parser.add_argument("--title", type=str, default='title', help="title")
parser.add_argument("--n", type=int, default=1000, help="n average")
args = parser.parse_args()
if __name__ == '__main__':
    with open(args.file_list, 'r') as file_list:
        for line in file_list:
            line = line.strip()
            rewards = []
            file = open(line, 'r')
            reward = 0
            count = 0
            for r in file:
                reward += float(r)
                if count % args.n == args.n - 1:
                    rewards.append(reward)
                    reward = 0
                else:
                    count += 1

            plt.plot(rewards, label = line)

    plt.title(args.title)
        
    plt.xlabel("eposides")
    plt.ylabel("rewards")

    plt.legend(loc='best')



    plt.show() 

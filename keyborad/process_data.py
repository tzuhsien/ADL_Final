from pickle import load, dump
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--n", type=int, default=10, help="n-step")
parser.add_argument("--input", type=str, default=None, help="input path")
parser.add_argument("--outout", type=str, default=None, help="output path")
args = parser.parse_args()


if __name__ == '__main__':
    f = open(args.input, 'rb')
    if args.output is not None:
        out_file = open(args.output. 'wb')
    memory = load(f)
    len = len(memory)
    for i in range(len):
        gamma = 1
        discount = 0.99
        reward = 0
        for j in range(args.n):
            if i + j >= len:
                break
            reward += memory[i + j][3] * gamma
            gamma *= discount
        memory[i].append(reward)
        memory[i].append(memory[i + args.n] if i + args.n < len else None) 

    if args.output is not None:
        dump(memory, out_file)

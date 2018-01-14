from pickle import load, dump
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--n", type=int, default=10, help="n-step")
parser.add_argument("--input", type=str, default=None, help="input path")
parser.add_argument("--output", type=str, default=None, help="output path")
args = parser.parse_args()

output = []
if __name__ == '__main__':
    f = open(args.input, 'r')
    if args.output is not None:
        out_file = open(args.output, 'wb')
    for line in f:
        line = line.strip()
        print (line)
        tmp = open(line, 'rb')
        memory = load(tmp)
        length = len(memory)
        for i in range(length):
            gamma = 1
            discount = 0.99
            reward = 0
            for j in range(args.n):
                if i + j >= length:
                    break
                reward += memory[i + j][3] * gamma
                gamma *= discount
            memory[i].append(reward)
            memory[i].append(memory[i + args.n][0] if i + args.n < length else None) 
        output += memory
        tmp.close()
    if args.output is not None:
        dump(output, out_file)

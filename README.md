# ADL Final Project

DQfD

## Prerequest

### Library

- OpenAI Gym Atari environment
- Pytorch 0.3

Please refer to [OpenAI's page](https://github.com/openai/gym) if you have any problem while installing.

### Demonstration Data

TODO

## How to run :

training dqfd:
* `$ python main.py --train --type dqfd --demo_file <demo_file> --env_name <env_name>`

testing dqfd:
* `$ python main.py --test --type dqfd --from-model <model_path> --env_name <env_name>`


# ADL Final Project

DQfD

## Prerequest

### System
- DISTRIB_ID=Ubuntu
- DISTRIB_RELEASE=16.04
- DISTRIB_CODENAME=xenial
- DISTRIB_DESCRIPTION="Ubuntu 16.04.2 LTS"
### Library
- OpenAI Gym Atari environment
- Pytorch 0.3

Please refer to [OpenAI's page](https://github.com/openai/gym) if you have any problem while installing.

### Demonstration Data

- linux10.csie.ntu.edu.tw:/tmp2/B03902080/<Enduro, Seaquest, SpaceInvaders>

## How to run :

training dqfd:
* `$ python main.py --train --type dqfd --demo_file <demo_file> --env_name <env_name>`

testing dqfd:
* `$ python main.py --test --type dqfd --model_path <model_path> --env_name <env_name>`

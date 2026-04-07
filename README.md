# maze_rl

Tiny maze RL project with:
- a headless maze environment written in C
- Python training via PufferLib
- a masked policy that uses valid-action bits from the env

## What This Repo Does

The maze is simulated in C for speed and exposed to Python through `ctypes`. Training happens in `py_train/train.py` using PufferLib and PyTorch.

The setup that worked best was:
- 4 actions: up, down, left, right
- observation = grid + normalized agent/goal coordinates + action mask
- simple distance-to-goal shaping
- imitation pretraining on the known shortest path
- PPO fine-tuning from the pretrained checkpoint

## Result

On the fixed maze, pretraining + PPO reached:
- `success_rate=100%`
- `mean_reward=13.96`
- `mean_length=27`

## Quick Start

Build the macOS shared library:

```bash
clang -O3 -dynamiclib -o c_env/maze_env.dylib c_env/maze_env.c
```

Install Python deps in `py_train`, then run:

```bash
cd py_train
python train.py --mode smoke --max-steps 128
python train.py --mode pretrain --max-steps 128 --eval-episodes 10
python train.py --mode train --init-checkpoint experiments/<timestamp>/pretrain.pt --num-envs 64 --max-steps 128 --total-timesteps 2000000 --eval-episodes 10 --eval-interval 100000
python train.py --mode eval --checkpoint latest --eval-episodes 3 --max-steps 128 --trace-episode
```

Replace `experiments/<timestamp>/pretrain.pt` with the path printed by the pretrain run.

## Notes

- PPO from scratch repeatedly found local loops on this maze.
- Action masking helped, but imitation pretraining was the step that made the policy reliably solve the task.


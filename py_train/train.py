from __future__ import annotations

import argparse
import glob
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pufferlib.emulation
import pufferlib.models
import pufferlib.pufferl
import pufferlib.pytorch
import pufferlib.vector
import torch
import torch.nn as nn

from maze_ctypes import DEFAULT_MAX_STEPS, OBSERVATION_SIZE, MazeBatchEnv


try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as exc:  # pragma: no cover - import guard for fresh envs
    raise SystemExit(
        "gymnasium is required. Install dependencies first: "
        "python -m pip install -r py_train/requirements.txt"
    ) from exc


@dataclass
class TinyTrainConfig:
    num_envs: int = 32
    max_steps: int = 48
    total_episodes: int = 10
    seed: int = 0
    mode: str = "smoke"
    total_timesteps: int = 100_000
    eval_episodes: int = 10
    checkpoint: str = "latest"
    init_checkpoint: str = ""
    eval_interval: int = 20_000
    trace_episode: bool = False
    pretrain_epochs: int = 200
    pretrain_repeats: int = 512
    pretrain_batch_size: int = 64


class MazeGymEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, max_steps: int = DEFAULT_MAX_STEPS):
        super().__init__()
        self.backend = MazeBatchEnv(num_envs=1, max_steps=max_steps)
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=-1.0,
            high=2.0,
            shape=(OBSERVATION_SIZE,),
            dtype=np.float32,
        )

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        obs = np.asarray(self.backend.reset()[0], dtype=np.float32)
        return obs, {}

    def step(self, action: int):
        observations, rewards, dones, truncateds = self.backend.step([int(action)])
        obs = np.asarray(observations[0], dtype=np.float32)
        reward = float(rewards[0])
        terminated = bool(dones[0])
        truncated = bool(truncateds[0])
        env_ref = self.backend.envs[0]
        success = bool(env_ref.playerX == env_ref.goalX and env_ref.playerY == env_ref.goalY)
        return obs, reward, terminated, truncated, {"success": success}


class MaskedMazePolicy(nn.Module):
    def __init__(self, env, hidden_size: int = 128):
        super().__init__()
        self.hidden_size = hidden_size
        self.obs_size = int(np.prod(env.single_observation_space.shape))
        self.action_count = env.single_action_space.n
        self.feature_size = self.obs_size - self.action_count

        self.encoder = nn.Sequential(
            nn.Linear(self.feature_size, hidden_size),
            nn.GELU(),
        )
        self.decoder = pufferlib.pytorch.layer_init(
            nn.Linear(hidden_size, self.action_count), std=0.01
        )
        self.value = pufferlib.pytorch.layer_init(
            nn.Linear(hidden_size, 1), std=1.0
        )

    def forward_eval(self, observations, state=None):
        return self.forward(observations, state)

    def forward(self, observations, state=None):
        features = observations[..., :self.feature_size]
        action_mask = observations[..., self.feature_size:]
        hidden = self.encoder(features)
        logits = self.decoder(hidden)
        masked_logits = logits.masked_fill(action_mask <= 0, -1e9)
        values = self.value(hidden)
        return masked_logits, values


def shortest_path_actions() -> list[int]:
    # Deterministic path from S to G in the fixed maze.
    script = "DDDDRRDDDRRURRUUUURRRDDLDDD"
    action_map = {"U": 0, "D": 1, "L": 2, "R": 3}
    return [action_map[action] for action in script]


def run_smoke_test(config: TinyTrainConfig) -> None:
    env = MazeGymEnv(max_steps=config.max_steps)
    scripted_actions = shortest_path_actions()

    for episode in range(config.total_episodes):
        obs, _ = env.reset(seed=config.seed + episode)
        total_reward = 0.0
        terminated = False
        truncated = False

        for step_idx, action in enumerate(scripted_actions, start=1):
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward

            if terminated or truncated:
                print(
                    f"episode={episode} step={step_idx} reward={total_reward:.2f} "
                    f"terminated={terminated} truncated={truncated} obs_sum={float(obs.sum()):.1f}"
                )
                break

        if not terminated and not truncated:
            print(
                f"episode={episode} step={len(scripted_actions)} reward={total_reward:.2f} "
                f"terminated={terminated} truncated={truncated} obs_sum={float(obs.sum()):.1f}"
            )


def make_pufferlib_env_creator(max_steps: int):
    def creator(buf=None, seed=0):
        return pufferlib.emulation.GymnasiumPufferEnv(
            env_creator=MazeGymEnv,
            env_kwargs={"max_steps": max_steps},
            buf=buf,
            seed=seed,
        )

    return creator


def maybe_load_checkpoint(policy: nn.Module, checkpoint: str) -> None:
    if not checkpoint:
        return

    checkpoint_path = resolve_checkpoint_path(checkpoint)
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    state_dict = {key.replace("module.", ""): value for key, value in state_dict.items()}
    policy.load_state_dict(state_dict)


def build_imitation_dataset(max_steps: int, repeats: int) -> tuple[torch.Tensor, torch.Tensor]:
    env = MazeGymEnv(max_steps=max_steps)
    observations = []
    actions = []
    scripted_actions = shortest_path_actions()

    for episode in range(repeats):
        obs, _ = env.reset(seed=episode)
        for action in scripted_actions:
            observations.append(obs.copy())
            actions.append(action)
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break

    obs_tensor = torch.tensor(np.asarray(observations, dtype=np.float32))
    action_tensor = torch.tensor(actions, dtype=torch.long)
    return obs_tensor, action_tensor


def save_policy_checkpoint(policy: nn.Module, prefix: str = "pretrain") -> Path:
    run_dir = Path("experiments") / str(int(time.time() * 1000))
    run_dir.mkdir(parents=True, exist_ok=True)
    model_path = run_dir / f"{prefix}.pt"
    torch.save(policy.state_dict(), model_path)
    return model_path


def run_pretrain(config: TinyTrainConfig) -> None:
    vecenv = pufferlib.vector.make(
        make_pufferlib_env_creator(max_steps=config.max_steps),
        backend=pufferlib.vector.Serial,
        num_envs=1,
        seed=config.seed,
    )
    policy = MaskedMazePolicy(vecenv, hidden_size=128)
    vecenv.close()

    observations, actions = build_imitation_dataset(
        max_steps=config.max_steps,
        repeats=config.pretrain_repeats,
    )
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)

    for epoch in range(config.pretrain_epochs):
        permutation = torch.randperm(len(actions))
        total_loss = 0.0

        for start in range(0, len(actions), config.pretrain_batch_size):
            indices = permutation[start : start + config.pretrain_batch_size]
            batch_obs = observations[indices]
            batch_actions = actions[indices]

            logits, _ = policy(batch_obs)
            loss = torch.nn.functional.cross_entropy(logits, batch_actions)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(indices)

        if (epoch + 1) % max(1, config.pretrain_epochs // 10) == 0 or epoch == 0:
            mean_loss = total_loss / len(actions)
            print(f"pretrain_epoch={epoch + 1} loss={mean_loss:.4f}")

    model_path = save_policy_checkpoint(policy, prefix="pretrain")
    print(f"saved_pretrain_checkpoint={model_path}")

    metrics = evaluate_policy(
        policy,
        max_steps=config.max_steps,
        eval_episodes=config.eval_episodes,
        seed=config.seed,
    )
    print(
        f"pretrain_eval success_rate={metrics['success_rate']:.2%} "
        f"mean_reward={metrics['mean_reward']:.2f} "
        f"mean_length={metrics['mean_length']:.1f}"
    )


def run_tiny_training(config: TinyTrainConfig) -> None:
    vecenv = pufferlib.vector.make(
        make_pufferlib_env_creator(max_steps=config.max_steps),
        backend=pufferlib.vector.Serial,
        num_envs=config.num_envs,
        seed=config.seed,
    )

    policy = MaskedMazePolicy(vecenv, hidden_size=128)
    maybe_load_checkpoint(policy, config.init_checkpoint)
    argv = sys.argv
    try:
        sys.argv = [sys.argv[0]]
        args = pufferlib.pufferl.load_config("default")
    finally:
        sys.argv = argv

    args["vec"]["num_envs"] = config.num_envs
    args["vec"]["seed"] = config.seed
    args["train"]["seed"] = config.seed
    args["train"]["device"] = "cpu"
    args["train"]["optimizer"] = "adam"
    args["train"]["compile"] = False
    args["train"]["cpu_offload"] = False
    args["train"]["total_timesteps"] = config.total_timesteps
    args["train"]["batch_size"] = "auto"
    args["train"]["bptt_horizon"] = 32
    batch_size = config.num_envs * args["train"]["bptt_horizon"]
    args["train"]["minibatch_size"] = batch_size
    args["train"]["max_minibatch_size"] = batch_size
    args["train"]["learning_rate"] = 3e-4
    args["train"]["checkpoint_interval"] = 10_000

    train_config = dict(**args["train"], env="default")
    logger = pufferlib.pufferl.NoLogger(args)
    trainer = pufferlib.pufferl.PuffeRL(train_config, vecenv, policy, logger)
    next_eval_step = config.eval_interval

    while trainer.global_step < train_config["total_timesteps"]:
        trainer.evaluate()
        trainer.train()

        if trainer.global_step >= next_eval_step:
            metrics = evaluate_policy(
                trainer.uncompiled_policy,
                max_steps=config.max_steps,
                eval_episodes=config.eval_episodes,
                seed=config.seed,
            )
            print(
                f"periodic_eval step={trainer.global_step} "
                f"success_rate={metrics['success_rate']:.2%} "
                f"mean_reward={metrics['mean_reward']:.2f} "
                f"mean_length={metrics['mean_length']:.1f}"
            )
            next_eval_step += config.eval_interval

    model_path = trainer.close()
    print(f"saved_checkpoint={model_path}")


def latest_checkpoint_path() -> Path:
    candidates = sorted(
        glob.glob("experiments/**/*.pt", recursive=True),
        key=lambda path: Path(path).stat().st_mtime,
    )
    candidates = [Path(path) for path in candidates if Path(path).name.startswith("model_")]
    if not candidates:
        raise FileNotFoundError("No checkpoint found under py_train/experiments")

    return candidates[-1]


def resolve_checkpoint_path(checkpoint: str) -> Path:
    if checkpoint == "latest":
        return latest_checkpoint_path()

    path = Path(checkpoint)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    return path


def run_eval(config: TinyTrainConfig) -> None:
    checkpoint_path = resolve_checkpoint_path(config.checkpoint)

    vecenv = pufferlib.vector.make(
        make_pufferlib_env_creator(max_steps=config.max_steps),
        backend=pufferlib.vector.Serial,
        num_envs=1,
        seed=config.seed,
    )
    policy = MaskedMazePolicy(vecenv, hidden_size=128)
    maybe_load_checkpoint(policy, str(checkpoint_path))
    policy.eval()
    vecenv.close()

    metrics = evaluate_policy(
        policy,
        max_steps=config.max_steps,
        eval_episodes=config.eval_episodes,
        seed=config.seed,
        trace_episode=config.trace_episode,
    )
    for idx, episode in enumerate(metrics["episodes"]):
        print(
            f"eval_episode={idx} reward={episode['reward']:.2f} "
            f"steps={episode['steps']} success={episode['success']} truncated={episode['truncated']}"
        )
        if config.trace_episode and idx == 0:
            trace = " ".join(
                f"[t={step} a={action} pos=({x},{y})]"
                for step, action, x, y in episode["trajectory"]
            )
            print(f"trace_episode_0={trace}")

    print(f"checkpoint={checkpoint_path}")
    print(f"success_rate={metrics['success_rate']:.2%}")
    print(f"mean_reward={metrics['mean_reward']:.2f}")
    print(f"mean_length={metrics['mean_length']:.1f}")


def evaluate_policy(
    policy,
    max_steps: int,
    eval_episodes: int,
    seed: int,
    trace_episode: bool = False,
) -> dict[str, object]:
    env = MazeGymEnv(max_steps=max_steps)
    successes = 0
    rewards = []
    lengths = []
    episodes = []

    with torch.no_grad():
        for episode in range(eval_episodes):
            obs, _ = env.reset(seed=seed + episode)
            total_reward = 0.0
            steps = 0
            terminated = False
            truncated = False
            trajectory = []

            while not (terminated or truncated):
                obs_tensor = torch.from_numpy(obs).unsqueeze(0)
                logits, _ = policy(obs_tensor)
                action = int(torch.argmax(logits, dim=-1).item())
                env_ref = env.backend.envs[0]
                trajectory.append((steps, action, int(env_ref.playerX), int(env_ref.playerY)))
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                steps += 1

            success = bool(info.get("success", False))
            successes += int(success)
            rewards.append(total_reward)
            lengths.append(steps)
            episodes.append(
                {
                    "reward": total_reward,
                    "steps": steps,
                    "success": success,
                    "truncated": truncated,
                    "trajectory": trajectory,
                }
            )

    return {
        "episodes": episodes,
        "success_rate": successes / eval_episodes,
        "mean_reward": sum(rewards) / len(rewards),
        "mean_length": sum(lengths) / len(lengths),
    }


def print_pufferlib_recipe(config: TinyTrainConfig) -> None:
    print("Tiny training recipe")
    print(f"- num_envs: {config.num_envs}")
    print(f"- max_steps: {config.max_steps}")
    print("- observation: 120 grid values + 4 coordinates + 4 action-mask values")
    print("- action space: 4 discrete moves")
    print("")
    print("Use this environment creator with PufferLib:")
    print("```python")
    print("import pufferlib.vector")
    print("from train import make_pufferlib_env_creator")
    print("")
    print("vecenv = pufferlib.vector.make(")
    print("    make_pufferlib_env_creator(max_steps=48),")
    print("    num_envs=32,")
    print(")")
    print("```")
    print("")
    print("Then attach `vecenv` to your `pufferl.PuffeRL(...)` trainer.")


def parse_args() -> TinyTrainConfig:
    parser = argparse.ArgumentParser(description="Tiny maze training scaffold")
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--max-steps", type=int, default=48)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--total-timesteps", type=int, default=100_000)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--eval-interval", type=int, default=20_000)
    parser.add_argument("--checkpoint", type=str, default="latest")
    parser.add_argument("--init-checkpoint", type=str, default="")
    parser.add_argument("--pretrain-epochs", type=int, default=200)
    parser.add_argument("--pretrain-repeats", type=int, default=512)
    parser.add_argument("--pretrain-batch-size", type=int, default=64)
    parser.add_argument("--trace-episode", action="store_true")
    parser.add_argument(
        "--mode",
        choices=["smoke", "recipe", "pretrain", "train", "eval"],
        default="smoke",
        help="`smoke` validates the env, `pretrain` does imitation learning, `train` runs PPO, `eval` tests a checkpoint, and `recipe` prints the hookup.",
    )
    args = parser.parse_args()
    return TinyTrainConfig(
        num_envs=args.num_envs,
        max_steps=args.max_steps,
        total_episodes=args.episodes,
        seed=args.seed,
        mode=args.mode,
        total_timesteps=args.total_timesteps,
        eval_episodes=args.eval_episodes,
        checkpoint=args.checkpoint,
        init_checkpoint=args.init_checkpoint,
        eval_interval=args.eval_interval,
        trace_episode=args.trace_episode,
        pretrain_epochs=args.pretrain_epochs,
        pretrain_repeats=args.pretrain_repeats,
        pretrain_batch_size=args.pretrain_batch_size,
    )


def main() -> None:
    config = parse_args()

    if config.mode == "smoke":
        run_smoke_test(config)
        return

    if config.mode == "train":
        run_tiny_training(config)
        return

    if config.mode == "pretrain":
        run_pretrain(config)
        return

    if config.mode == "eval":
        run_eval(config)
        return

    print_pufferlib_recipe(config)


if __name__ == "__main__":
    main()

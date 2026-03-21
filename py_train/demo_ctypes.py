from maze_ctypes import ACTION_DOWN, ACTION_RIGHT, ACTION_STAY, MazeBatchEnv


def main() -> None:
    env = MazeBatchEnv(num_envs=3)
    observations = env.reset()
    print(f"reset observation shape = ({len(observations)}, {len(observations[0])})")

    actions = [ACTION_RIGHT, ACTION_DOWN, ACTION_STAY]
    observations, rewards, dones, truncateds = env.step(actions)

    print(f"step observation shape = ({len(observations)}, {len(observations[0])})")
    for index, reward in enumerate(rewards):
        print(
            f"env={index} action={actions[index]} reward={reward:.2f} "
            f"done={dones[index]} truncated={truncateds[index]}"
        )


if __name__ == "__main__":
    main()

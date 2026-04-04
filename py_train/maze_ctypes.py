from __future__ import annotations

import ctypes
import sys
from pathlib import Path


GRID_WIDTH = 12
GRID_HEIGHT = 10
OBSERVATION_SIZE = GRID_WIDTH * GRID_HEIGHT + 8
DEFAULT_MAX_STEPS = 128

ACTION_UP = 0
ACTION_DOWN = 1
ACTION_LEFT = 2
ACTION_RIGHT = 3


class MazeStepResult(ctypes.Structure):
    _fields_ = [
        ("reward", ctypes.c_float),
        ("done", ctypes.c_bool),
        ("truncated", ctypes.c_bool),
    ]


class MazeEnv(ctypes.Structure):
    _fields_ = [
        ("playerX", ctypes.c_int),
        ("playerY", ctypes.c_int),
        ("startX", ctypes.c_int),
        ("startY", ctypes.c_int),
        ("goalX", ctypes.c_int),
        ("goalY", ctypes.c_int),
        ("steps", ctypes.c_int),
        ("maxSteps", ctypes.c_int),
        ("episodeReward", ctypes.c_float),
        ("lastReward", ctypes.c_float),
        ("done", ctypes.c_bool),
        ("truncated", ctypes.c_bool),
        ("renderEnabled", ctypes.c_bool),
    ]


def _default_library_path() -> Path:
    root = Path(__file__).resolve().parent.parent / "c_env"

    if sys.platform == "win32":
        candidates = [root / "maze_env.dll"]
    elif sys.platform == "darwin":
        candidates = [root / "maze_env.dylib", root / "maze_env.so"]
    else:
        candidates = [root / "maze_env.so", root / "maze_env.dylib"]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return candidates[0]


class MazeLibrary:
    def __init__(self, dll_path: Path | str | None = None) -> None:
        library_path = Path(dll_path) if dll_path is not None else _default_library_path()
        self.lib = ctypes.CDLL(str(library_path))

        self.lib.maze_init_env.argtypes = [ctypes.POINTER(MazeEnv), ctypes.c_bool, ctypes.c_int]
        self.lib.maze_init_env.restype = None

        self.lib.reset.argtypes = [ctypes.POINTER(MazeEnv)]
        self.lib.reset.restype = None

        self.lib.reset_batch.argtypes = [ctypes.POINTER(MazeEnv), ctypes.c_int]
        self.lib.reset_batch.restype = None

        self.lib.step.argtypes = [ctypes.POINTER(MazeEnv), ctypes.c_int]
        self.lib.step.restype = MazeStepResult

        self.lib.step_batch.argtypes = [
            ctypes.POINTER(MazeEnv),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
        ]
        self.lib.step_batch.restype = None

        self.lib.get_observation.argtypes = [ctypes.POINTER(MazeEnv), ctypes.POINTER(ctypes.c_float)]
        self.lib.get_observation.restype = None

        self.lib.get_observation_batch.argtypes = [
            ctypes.POINTER(MazeEnv),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
        ]
        self.lib.get_observation_batch.restype = None


class MazeBatchEnv:
    def __init__(self, num_envs: int, max_steps: int = DEFAULT_MAX_STEPS, dll_path: Path | str | None = None) -> None:
        if num_envs <= 0:
            raise ValueError("num_envs must be positive")

        self.num_envs = num_envs
        self.lib = MazeLibrary(dll_path)
        self.envs = (MazeEnv * num_envs)()
        self.actions = (ctypes.c_int * num_envs)()
        self.rewards = (ctypes.c_float * num_envs)()
        self.dones = (ctypes.c_int * num_envs)()
        self.truncateds = (ctypes.c_int * num_envs)()
        self.observations = (ctypes.c_float * (num_envs * OBSERVATION_SIZE))()

        for i in range(num_envs):
            self.lib.lib.maze_init_env(ctypes.byref(self.envs[i]), False, max_steps)

    def reset(self) -> list[list[float]]:
        self.lib.lib.reset_batch(self.envs, self.num_envs)
        return self.get_observation()

    def reset_index(self, env_index: int) -> list[float]:
        if env_index < 0 or env_index >= self.num_envs:
            raise IndexError(f"env_index out of range: {env_index}")

        self.lib.lib.reset(ctypes.byref(self.envs[env_index]))
        self.lib.lib.get_observation(
            ctypes.byref(self.envs[env_index]),
            ctypes.byref(self.observations, env_index * OBSERVATION_SIZE * ctypes.sizeof(ctypes.c_float)),
        )

        start = env_index * OBSERVATION_SIZE
        return [float(self.observations[start + cell]) for cell in range(OBSERVATION_SIZE)]

    def step(self, actions: list[int] | tuple[int, ...]) -> tuple[list[list[float]], list[float], list[int], list[int]]:
        if len(actions) != self.num_envs:
            raise ValueError(f"expected {self.num_envs} actions, got {len(actions)}")

        for i, action in enumerate(actions):
            self.actions[i] = int(action)

        self.lib.lib.step_batch(
            self.envs,
            self.actions,
            self.rewards,
            self.dones,
            self.truncateds,
            self.num_envs,
        )

        observations = self.get_observation()
        rewards = [float(self.rewards[i]) for i in range(self.num_envs)]
        dones = [int(self.dones[i]) for i in range(self.num_envs)]
        truncateds = [int(self.truncateds[i]) for i in range(self.num_envs)]
        return observations, rewards, dones, truncateds

    def get_observation(self) -> list[list[float]]:
        self.lib.lib.get_observation_batch(self.envs, self.observations, self.num_envs)
        return [
            [float(self.observations[(env_index * OBSERVATION_SIZE) + cell]) for cell in range(OBSERVATION_SIZE)]
            for env_index in range(self.num_envs)
        ]

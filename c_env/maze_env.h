#ifndef MAZE_ENV_H
#define MAZE_ENV_H

#include <stdbool.h>

#ifdef _WIN32
#define MAZE_API __declspec(dllexport)
#else
#define MAZE_API
#endif

#define GRID_WIDTH 12
#define GRID_HEIGHT 10
#define OBSERVATION_SIZE (GRID_WIDTH * GRID_HEIGHT + 8)
#define DEFAULT_MAX_STEPS 128

typedef enum MazeAction {
    ACTION_UP = 0,
    ACTION_DOWN = 1,
    ACTION_LEFT = 2,
    ACTION_RIGHT = 3
} MazeAction;

typedef struct MazeStepResult {
    float reward;
    bool done;
    bool truncated;
} MazeStepResult;

typedef struct MazeEnv {
    int playerX;
    int playerY;
    int startX;
    int startY;
    int goalX;
    int goalY;
    int steps;
    int maxSteps;
    float episodeReward;
    float lastReward;
    bool done;
    bool truncated;
    bool renderEnabled;
} MazeEnv;

MAZE_API void maze_init_env(MazeEnv *env, bool renderEnabled, int maxSteps);
MAZE_API void reset(MazeEnv *env);
MAZE_API void reset_batch(MazeEnv *envs, int numEnvs);
MAZE_API MazeStepResult step(MazeEnv *env, MazeAction action);
MAZE_API void step_batch(
    MazeEnv *envs,
    const int *actions,
    float *rewards,
    int *dones,
    int *truncateds,
    int numEnvs
);
MAZE_API void get_observation(const MazeEnv *env, float *outBuffer);
MAZE_API void get_observation_batch(const MazeEnv *envs, float *outBuffer, int numEnvs);
MAZE_API const char *maze_action_name(MazeAction action);
MAZE_API MazeAction maze_parse_action_char(int c);

#endif

#include "maze_env.h"

#include <string.h>

static const char *kMaze[GRID_HEIGHT] = {
    "############",
    "#S...#.....#",
    "#.#.#.###..#",
    "#.#...#....#",
    "#.###.#.##.#",
    "#.....#.#..#",
    "###.###.#.##",
    "#...#...#..#",
    "#.#...###G.#",
    "############"
};

static void FindTile(char tileType, int *outX, int *outY)
{
    for (int y = 0; y < GRID_HEIGHT; y++) {
        for (int x = 0; x < GRID_WIDTH; x++) {
            if (kMaze[y][x] == tileType) {
                *outX = x;
                *outY = y;
                return;
            }
        }
    }

    *outX = 0;
    *outY = 0;
}

static bool IsWall(int x, int y)
{
    if (x < 0 || x >= GRID_WIDTH || y < 0 || y >= GRID_HEIGHT) {
        return true;
    }

    return kMaze[y][x] == '#';
}

void maze_init_env(MazeEnv *env, bool renderEnabled, int maxSteps)
{
    memset(env, 0, sizeof(*env));
    FindTile('S', &env->startX, &env->startY);
    FindTile('G', &env->goalX, &env->goalY);
    env->renderEnabled = renderEnabled;
    env->maxSteps = (maxSteps > 0) ? maxSteps : DEFAULT_MAX_STEPS;
}

void reset(MazeEnv *env)
{
    env->playerX = env->startX;
    env->playerY = env->startY;
    env->steps = 0;
    env->episodeReward = 0.0f;
    env->lastReward = 0.0f;
    env->done = false;
    env->truncated = false;
}

void reset_batch(MazeEnv *envs, int numEnvs)
{
    for (int i = 0; i < numEnvs; i++) {
        reset(&envs[i]);
    }
}

MazeStepResult step(MazeEnv *env, MazeAction action)
{
    MazeStepResult result = {0};

    if (env->done || env->truncated) {
        result.done = env->done;
        result.truncated = env->truncated;
        return result;
    }

    int dx = 0;
    int dy = 0;

    switch (action) {
        case ACTION_UP:
            dy = -1;
            break;
        case ACTION_DOWN:
            dy = 1;
            break;
        case ACTION_LEFT:
            dx = -1;
            break;
        case ACTION_RIGHT:
            dx = 1;
            break;
        case ACTION_STAY:
        default:
            break;
    }

    int nextX = env->playerX + dx;
    int nextY = env->playerY + dy;
    bool blocked = IsWall(nextX, nextY);

    result.reward = -0.1f;

    if (action == ACTION_STAY) {
        result.reward = -0.2f;
    } else if (blocked) {
        result.reward = -0.5f;
    } else {
        env->playerX = nextX;
        env->playerY = nextY;
    }

    env->steps += 1;

    if (env->playerX == env->goalX && env->playerY == env->goalY) {
        result.reward += 10.0f;
        env->done = true;
    } else if (env->steps >= env->maxSteps) {
        env->truncated = true;
    }

    env->lastReward = result.reward;
    env->episodeReward += result.reward;
    result.done = env->done;
    result.truncated = env->truncated;
    return result;
}

void step_batch(
    MazeEnv *envs,
    const int *actions,
    float *rewards,
    int *dones,
    int *truncateds,
    int numEnvs
)
{
    for (int i = 0; i < numEnvs; i++) {
        MazeStepResult result = step(&envs[i], (MazeAction)actions[i]);
        rewards[i] = result.reward;
        dones[i] = result.done ? 1 : 0;

        if (truncateds != NULL) {
            truncateds[i] = result.truncated ? 1 : 0;
        }
    }
}

void get_observation(const MazeEnv *env, float *outBuffer)
{
    for (int y = 0; y < GRID_HEIGHT; y++) {
        for (int x = 0; x < GRID_WIDTH; x++) {
            int index = y * GRID_WIDTH + x;
            float value = 0.0f;

            if (kMaze[y][x] == '#') {
                value = -1.0f;
            } else if (x == env->goalX && y == env->goalY) {
                value = 1.0f;
            }

            if (x == env->playerX && y == env->playerY) {
                value = 2.0f;
            }

            outBuffer[index] = value;
        }
    }
}

void get_observation_batch(const MazeEnv *envs, float *outBuffer, int numEnvs)
{
    for (int i = 0; i < numEnvs; i++) {
        get_observation(&envs[i], outBuffer + (i * OBSERVATION_SIZE));
    }
}

const char *maze_action_name(MazeAction action)
{
    switch (action) {
        case ACTION_UP:
            return "up";
        case ACTION_DOWN:
            return "down";
        case ACTION_LEFT:
            return "left";
        case ACTION_RIGHT:
            return "right";
        case ACTION_STAY:
            return "stay";
        default:
            return "unknown";
    }
}

MazeAction maze_parse_action_char(int c)
{
    switch (c) {
        case 'w':
            return ACTION_UP;
        case 's':
            return ACTION_DOWN;
        case 'a':
            return ACTION_LEFT;
        case 'd':
            return ACTION_RIGHT;
        default:
            return ACTION_STAY;
    }
}

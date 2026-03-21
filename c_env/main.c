#include "maze_env.h"

#include <stdbool.h>
#include <stdio.h>
#include <string.h>

#ifndef MAZE_HEADLESS
#include "raylib.h"
#endif

#define TILE_SIZE 48
#define HUD_HEIGHT 96
#define SCREEN_WIDTH (GRID_WIDTH * TILE_SIZE)
#define SCREEN_HEIGHT (GRID_HEIGHT * TILE_SIZE + HUD_HEIGHT)

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

static void PrintObservation(const float *buffer)
{
    for (int y = 0; y < GRID_HEIGHT; y++) {
        for (int x = 0; x < GRID_WIDTH; x++) {
            printf("%4.0f", buffer[y * GRID_WIDTH + x]);
        }
        printf("\n");
    }
}

static void RunHeadlessDemo(MazeEnv *env)
{
    const char *script = "ddddssddssdddd";
    float observation[OBSERVATION_SIZE] = {0};

    printf("Running headless environment demo\n");
    printf("Observation shape: [%d]\n", OBSERVATION_SIZE);

    reset(env);
    get_observation(env, observation);
    PrintObservation(observation);

    for (int i = 0; script[i] != '\0'; i++) {
        MazeAction action = maze_parse_action_char(script[i]);
        MazeStepResult result = step(env, action);

        printf(
            "step=%d action=%s reward=%.2f done=%d truncated=%d pos=(%d,%d)\n",
            env->steps,
            maze_action_name(action),
            result.reward,
            result.done ? 1 : 0,
            result.truncated ? 1 : 0,
            env->playerX,
            env->playerY
        );

        if (result.done || result.truncated) {
            break;
        }
    }

    get_observation(env, observation);
    PrintObservation(observation);
    printf("episode_reward=%.2f\n", env->episodeReward);
}

static void RunBatchDemo(bool renderEnabled)
{
    enum { BATCH_SIZE = 3 };

    MazeEnv envs[BATCH_SIZE];
    float rewards[BATCH_SIZE] = {0};
    int dones[BATCH_SIZE] = {0};
    int truncateds[BATCH_SIZE] = {0};
    int actions[BATCH_SIZE] = {ACTION_RIGHT, ACTION_DOWN, ACTION_STAY};
    float observations[BATCH_SIZE * OBSERVATION_SIZE] = {0};

    for (int i = 0; i < BATCH_SIZE; i++) {
        maze_init_env(&envs[i], renderEnabled, DEFAULT_MAX_STEPS);
    }

    reset_batch(envs, BATCH_SIZE);
    step_batch(envs, actions, rewards, dones, truncateds, BATCH_SIZE);
    get_observation_batch(envs, observations, BATCH_SIZE);

    printf("Running batched headless demo\n");
    for (int i = 0; i < BATCH_SIZE; i++) {
        printf(
            "env=%d action=%s reward=%.2f done=%d truncated=%d pos=(%d,%d)\n",
            i,
            maze_action_name((MazeAction)actions[i]),
            rewards[i],
            dones[i],
            truncateds[i],
            envs[i].playerX,
            envs[i].playerY
        );
    }

    printf("batch_observation_shape=[%d, %d]\n", BATCH_SIZE, OBSERVATION_SIZE);
}

#ifndef MAZE_HEADLESS
static MazeAction ReadPlayerAction(void)
{
    if (IsKeyPressed(KEY_UP) || IsKeyPressed(KEY_W)) {
        return ACTION_UP;
    }
    if (IsKeyPressed(KEY_DOWN) || IsKeyPressed(KEY_S)) {
        return ACTION_DOWN;
    }
    if (IsKeyPressed(KEY_LEFT) || IsKeyPressed(KEY_A)) {
        return ACTION_LEFT;
    }
    if (IsKeyPressed(KEY_RIGHT) || IsKeyPressed(KEY_D)) {
        return ACTION_RIGHT;
    }

    return ACTION_STAY;
}

static void DrawMaze(const MazeEnv *env)
{
    for (int y = 0; y < GRID_HEIGHT; y++) {
        for (int x = 0; x < GRID_WIDTH; x++) {
            Rectangle tileRect = {
                (float)(x * TILE_SIZE),
                (float)(y * TILE_SIZE),
                (float)TILE_SIZE,
                (float)TILE_SIZE
            };

            Color tileColor = (kMaze[y][x] == '#') ? DARKGRAY : (Color){225, 232, 214, 255};
            DrawRectangleRec(tileRect, tileColor);
            DrawRectangleLinesEx(tileRect, 1.0f, (Color){110, 120, 110, 80});

            if (x == env->goalX && y == env->goalY) {
                DrawRectangle(
                    x * TILE_SIZE + 10,
                    y * TILE_SIZE + 10,
                    TILE_SIZE - 20,
                    TILE_SIZE - 20,
                    GOLD
                );
            }

            if (x == env->playerX && y == env->playerY) {
                DrawCircle(
                    x * TILE_SIZE + TILE_SIZE / 2,
                    y * TILE_SIZE + TILE_SIZE / 2,
                    TILE_SIZE * 0.28f,
                    BLUE
                );
            }
        }
    }
}

static void DrawHud(const MazeEnv *env)
{
    DrawRectangle(0, GRID_HEIGHT * TILE_SIZE, SCREEN_WIDTH, HUD_HEIGHT, (Color){28, 37, 48, 255});
    DrawText(TextFormat("Steps: %d / %d", env->steps, env->maxSteps), 20, GRID_HEIGHT * TILE_SIZE + 16, 24, RAYWHITE);
    DrawText(TextFormat("Reward: %.2f", env->episodeReward), 20, GRID_HEIGHT * TILE_SIZE + 50, 24, RAYWHITE);

    const char *statusText = "Move with arrows/WASD. R resets the episode.";
    Color statusColor = LIGHTGRAY;

    if (env->done) {
        statusText = "Goal reached. Press R to reset.";
        statusColor = GREEN;
    } else if (env->truncated) {
        statusText = "Episode truncated at max steps. Press R to reset.";
        statusColor = ORANGE;
    }

    DrawText(statusText, 240, GRID_HEIGHT * TILE_SIZE + 34, 22, statusColor);
}

static void RunInteractive(MazeEnv *env)
{
    InitWindow(SCREEN_WIDTH, SCREEN_HEIGHT, "Maze RL Environment");
    SetTargetFPS(60);
    reset(env);

    while (!WindowShouldClose()) {
        if (IsKeyPressed(KEY_R)) {
            reset(env);
        } else {
            MazeAction action = ReadPlayerAction();
            if (action != ACTION_STAY) {
                step(env, action);
            }
        }

        BeginDrawing();
        ClearBackground((Color){245, 247, 240, 255});
        DrawMaze(env);
        DrawHud(env);
        EndDrawing();
    }

    CloseWindow();
}
#endif

int main(int argc, char **argv)
{
    bool renderEnabled = true;
    bool batchDemo = false;
    MazeEnv env = {0};

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--headless") == 0) {
            renderEnabled = false;
        } else if (strcmp(argv[i], "--batch-demo") == 0) {
            batchDemo = true;
            renderEnabled = false;
        }
    }

    maze_init_env(&env, renderEnabled, DEFAULT_MAX_STEPS);

#ifdef MAZE_HEADLESS
    (void)argc;
    (void)argv;
    if (batchDemo) {
        RunBatchDemo(renderEnabled);
    } else {
        RunHeadlessDemo(&env);
    }
#else
    if (batchDemo) {
        RunBatchDemo(renderEnabled);
    } else if (env.renderEnabled) {
        RunInteractive(&env);
    } else {
        RunHeadlessDemo(&env);
    }
#endif

    return 0;
}

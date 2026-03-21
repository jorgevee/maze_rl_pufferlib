$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$gcc = "C:\msys64\mingw64\bin\gcc.exe"
$includeDir = "C:\msys64\mingw64\include"
$libDir = "C:\msys64\mingw64\lib"

& $gcc "$PSScriptRoot\maze_env.c" "$PSScriptRoot\main.c" -I"$includeDir" -L"$libDir" -lraylib -lopengl32 -lgdi32 -lwinmm -o "$PSScriptRoot\maze_game.exe"
& $gcc -DMAZE_HEADLESS "$PSScriptRoot\maze_env.c" "$PSScriptRoot\main.c" -o "$PSScriptRoot\maze_headless.exe"
& $gcc -shared -DMAZE_HEADLESS "$PSScriptRoot\maze_env.c" -o "$PSScriptRoot\maze_env.dll"

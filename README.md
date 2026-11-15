# zinedine: RL tooling for Soccer Physics

This project is a BepInEx plugin (**SoccerRL**) plus a Python package (**soccer_rl**) that turn
*Soccer Physics* into a reinforcement-learning environment:

- **Headless**: `-batchmode -nographics`, no window and no GPU needed.
- **Accelerated and lockstep**: the game advances only when you ask it to, as fast as the CPU allows.
  That's about 2,500 physics ticks/s per process on a laptop (≈50× real time), and you can run one
  process per core.
- **Deterministic**: the same seed and the same actions give bit-identical trajectories.
- **Full state access**: positions, angles and velocities of every rigid body, plus scores, events
  and flags each step. You can also get or set any field, property or PlayerPref by path.
- **Programmatic control**: press or release any of the 4 players' buttons (P1, P1 goalie, P2, P2 goalie).
  The other team can be driven by the built-in AI, by a policy (self-play), or by a human on the keyboard.
- **Gymnasium env**: egocentric mirrored observations, so one policy plays either side. Supports
  reward shaping, match modifiers (rain, beach ball, …), vectorised envs and `rgb_array` rendering.

> **Not affiliated with Soccer Physics or its author.** This repository contains only our own code.
> It doesn't include the game or any of its files: you need your own copy of the game (Linux build).

```
SoccerPhysics/                 your copy of the game (not in the repo); BepInEx + the plugin get installed here
plugin/SoccerRL/               C# plugin source (Bridge.cs = lockstep server, Patches.cs, Introspect.cs)
soccer_rl/                     Python: client.py (protocol), launcher.py (processes), env.py (Gymnasium),
                               duel.py (two-player / frozen-opponent envs), selfplay.py, noise.py
examples/                      quickstart, benchmark, watch / play-against, PPO, self-play, ES, Sample Factory
scripts/                       install BepInEx, build the plugin, export / inspect Sample Factory runs
runs/*.sh                      launch scripts of the long training runs (see HANDOFF.md)
```

## Setup

1. Put your copy of the game's Linux build in `./SoccerPhysics/` (or point the scripts and
   `SOCCER_PHYSICS_DIR` at it).
2. Install BepInEx (downloads the official release into your game folder) and build the plugin:

```bash
scripts/install_bepinex.sh              # BepInEx 5.4.23.6 from github.com/BepInEx/BepInEx
scripts/build_plugin.sh                 # needs the .NET SDK, or nix (uses `nix shell nixpkgs#dotnet-sdk_8`)
```

   The plugin compiles against the game's own assemblies in your install (`SoccerPhysics_Data/Managed`);
   none of them are redistributed.

3. Python:

```bash
uv venv && uv pip install -e .          # numpy + gymnasium
uv pip install torch --index-url https://download.pytorch.org/whl/cpu && uv pip install -e '.[train]'   # optional: SB3
# Sample Factory (long runs) needs its own venv: it pins numpy 1.26 / gymnasium 0.29
uv venv .venv-sf --python 3.12 && VIRTUAL_ENV=.venv-sf uv pip install torch sample-factory -e .
```

On NixOS the Unity binary runs through `steam-run`. The launcher does this automatically when
`/etc/NIXOS` exists; override with `SOCCER_RL_WRAPPER="..."` (an empty string means no wrapper).
The game directory defaults to `./SoccerPhysics`; override with `SOCCER_PHYSICS_DIR`.

Without `-rl-port` the plugin stays inactive, so launching the game normally is unaffected.

## Usage

```python
from soccer_rl import SoccerPhysicsEnv

env = SoccerPhysicsEnv(opponent="builtin", frame_skip=4, episode="goal", shaping=0.05)
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(1)   # 1 = hold the button, 0 = release
env.close()
```

| Example | |
|---|---|
| `python examples/quickstart.py` | low-level client: stepping, reading and writing variables, self-play |
| `python examples/benchmark.py --procs 1 4` | throughput |
| `python examples/watch.py [--model runs/ppo_soccer.zip] [--speed 4]` | watch in a window |
| `python examples/watch.py --model runs/ppo_soccer.zip --human` | play P1 (Up arrow) against the agent |
| `python examples/train_ppo.py --envs 4 --steps 2000000` | PPO vs the built-in AI |

### Environment options (`SoccerPhysicsEnv`)

| arg | default | |
|---|---|---|
| `side` | `"p1"` | `p1` = blue (right), `p2` = red (left) |
| `opponent` | `"builtin"` | `builtin` (game AI, P2 only), `idle`, `random`, `keyboard`, or `callable(obs) -> action` |
| `two_buttons` | `False` | `Discrete(2)`, or `MultiBinary(2)` = [outfield player, goalie] |
| `frame_skip` | 4 | physics ticks per step (50 ticks = 1 s of real time) |
| `episode` | `"goal"` | `goal`: terminate at the first goal. `match`: first to 5 (reward ±1 per goal) |
| `end_on_out_of_bounds` | `True` | truncate a `goal` episode when the ball leaves the arena |
| `max_steps` | 1500 | truncation |
| `modifiers` | `"none"` | `none`, `random` (the game's own odds), or a dict, see below |
| `shaping` | 0 | dense reward = shaping × ball progress towards the opponent goal |
| `render_mode` | `None` | `human` (real-time window) or `rgb_array` (`env.render()` → H×W×3) |

Modifiers: `ground` (1 normal, 2 slippery), `splits` (acrobatic), `mini_goals`, `big_goals`,
`ball` (0 soccer, 1 american, 2 beach), `headless` (no heads).

**Observation (187 floats).** Bodies come in egocentric order: own player, own goalie, opponent
player, opponent goalie (6 parts each: Torso, Head, ArmB, Leg, LegKick, ArmF), then the ball.
Each body contributes `x/5, y/5, sin θ, cos θ, vx/10, vy/10, ω/500` = 175 values. After that
come `grounded, jumped, button` for the 4 players = 12 values. When playing P2 the x axis is
mirrored, so the observing team always attacks towards −x. `env.state` gives the raw, un-mirrored state.

**Vectorised:** `soccer_rl.make_vec_env(8, **env_kwargs)` (Gymnasium) or SB3's `make_vec_env`
with `SubprocVecEnv`. Each env owns one game process.

### Low-level client

```python
from soccer_rl import GameProcess, P1, P1G
with GameProcess(headless=True) as g:
    c = g.client
    s = c.reset(seed=1, ai=1, ball=2)            # State: bodies (25×6), players (4×4), scores, events, flags
    s = c.step(buttons=0b0011, mask=0b0011, repeat=8)
    c.get("obj:P1:Player.grounded", "pref:P2Score", "static:UnityEngine.Time.timeScale")
    c.set_values({"static:UnityEngine.Physics2D.gravity": (0, -5), "obj:P1:Player.jumpPower2": 900})
    print(c.tree())                              # scene hierarchy with components
    img = c.pixels(320, 180)                     # windowed mode only
```

Query syntax: `pref:Key`, `static:Namespace.Type.member`, `obj:GameObject/Path:Component.member[.sub]`.
Private members work, and `member()` calls a zero-argument method. `c.tree()` lists the scene
hierarchy and components, which is a good place to start looking for variables.

## How it works

- **Lockstep.** The plugin inserts a system at the start of Unity's `FixedUpdate` player-loop
  phase, so it runs before any game script and before the physics step. When a step finishes,
  it sends the state and blocks the main thread on the socket until the next command arrives.
  `Time.captureDeltaTime` makes every frame advance a fixed amount of game time, so wall-clock
  time never leaks in. Since the game always sets `fixedDeltaTime = timeScale × 0.02`, one
  frame equals `ticks_per_frame` physics ticks even during slow motion.
- **Actions** call the game's own `Player.pressKey()` every tick while held, and
  `releaseKey()` on release, exactly as the keyboard handler does.
- **Resets** set the PlayerPrefs the game reads (scores, AI, modes, modifiers), seed
  `UnityEngine.Random` when the scene loads, and reload `SceneMain`.
- **Patches** (active only with `-rl-port`):
  - `fast_restart` skips the slow-motion replay after goals and the out-of-bounds pause.
  - `fast_spawn` skips the 0.7 s blinking-ball intro.
  - `mute` silences audio.
  - Modifier overrides are re-applied after `Game.Start` re-rolls them.

  Toggle these with `client.configure(...)` or as reset options.
- **Events** (goal by P1 or P2, out of bounds, scene reload, game over) are detected from
  PlayerPrefs changes. They accumulate over the ticks of a step.

### Wire protocol

Little-endian TCP. Client → game: an opcode byte followed by arguments.

| op | args | reply |
|---|---|---|
| `S` | u8 buttons, u8 mask, u16 repeat | State |
| `R` | str options (`k=v;k=v`) | State |
| `O` | | State |
| `I` | | str: schema, settings, prefs |
| `G` / `X` | str queries / `path=value` lines | str: one line per query |
| `C` | str settings | str |
| `P` | u16 w, u16 h | u32 n, u16 w, u16 h, RGB24 bytes (bottom-up) |
| `T` | | str: scene tree |
| `Q` | | (game quits) |

`str` is a u32 length followed by UTF-8. In `buttons` and `mask`, bit *i* is player *i*
(0 = P1, 1 = P1G, 2 = P2, 3 = P2G). State is a u32 length followed by
`i64 tick, i32 episode_tick, i32 p1_score, i32 p2_score, i32 events, i32 flags, f32 time_scale,
f32 fixed_dt, i32 n_bodies, n_bodies × 6 f32, 4 × 4 f32`. See `Bridge.cs`.

Game command-line flags: `-rl-port N`, `-rl-host`, `-rl-no-wait`, `-rl-exit-on-disconnect`,
`-rl-ticks-per-frame N`, `-rl-fps N`. Setting `SOCCER_RL_DEBUG=1` logs tick diagnostics.

## Notes and gotchas

- **Throughput** is bounded by the game's per-tick cost (≈0.35 ms: scripts, physics, PlayerPrefs
  lookups), not by the socket (≈50 µs per round trip). A scene reload after each goal or reset
  costs ≈15 ms. Scale with more processes. The launcher passes `-job-worker-count 0` because
  Unity's idle job workers otherwise spin and steal CPU from neighbouring instances.
- **Pixels** need a graphics context. On a desktop just use `headless=False`. On a server, run
  under `xvfb-run -s "-screen 0 1280x720x24"`; Mesa llvmpipe works at ≈185 steps/s with 480×270
  captures. The score digits render as blank boxes in captures (a tk2d font quirk).
- `Application.runInBackground` is forced on. Otherwise an unfocused window (always the case
  under Xvfb) silently stops the game.
- `steam-run` gives the game a private `/tmp`, so the launcher captures the player log from
  stdout instead of using `-logFile <path>`. Pass `log_path=` to `GameProcess` to keep it.
- Each launched instance gets its own `XDG_CONFIG_HOME`, so PlayerPrefs never collide between
  instances and never touch your real save.
- The built-in AI only exists for P2 (`InputReceiver` checks `!isP1`). It presses at random,
  and more often when the ball is inside its trigger zones.
